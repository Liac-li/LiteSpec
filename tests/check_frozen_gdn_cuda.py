"""Read-only-on-disk real-model preflight: native state parity, identity, gradients."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from litespec.vlm import train_aligned as aligned
from litespec.vlm.benchmark import RequestPositionState
from litespec.vlm.gdn_capture import GDNAnchorCapture
from litespec.vlm.gdn_reader import cache_reader_states
from litespec.vlm.reader_adapt import initialize_reader_adaptation, frozen_digest
from litespec.vlm.runtime import load_target, prepare_example, read_records, configure_image_token_budget
from diagnose_gdn import reader_mode
from gdn_preflight_metrics import (state_pair_metrics, read_pair_metrics, STATE_LIMITS, READ_LIMIT,
                                  GDNInputCapture, input_pair_metrics, entry_gate)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',required=True);p.add_argument('--draft',required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.manual_seed(42)
    target,processor=load_target(args.target,device='cuda',dtype=torch.bfloat16,attention='sdpa')
    configure_image_token_budget(processor,target.config,1024)
    row=next(iter(read_records(args.data,1)))
    example=prepare_example(processor,target.config,row,device='cuda',training=True,max_length=4096,base_dir=args.data.parent)
    cfg=aligned.load_config(Path(__file__).resolve().parents[1]/'scripts/config/vlm/frozen_reader.yaml')
    cfg.update(pretrained_draft=args.draft,target_model=args.target,num_anchors=2)
    draft=initialize_reader_adaptation(target,cfg,attention='sdpa')
    reader=draft.layers[draft.config.gdn_reader_config['injection_layer']].gdn_reader
    length=example.inputs['input_ids'].shape[1]
    anchors=torch.tensor([[example.prefill_length,min(example.prefill_length+2,length-1)]],device='cuda')
    anchors=anchors.unique(sorted=True).view(1,-1);keep=torch.ones_like(anchors,dtype=torch.bool)
    with torch.no_grad(),RequestPositionState(target).activate():
        with GDNInputCapture(target,reader.layer_ids) as full_input_capture, GDNAnchorCapture(target,reader.layer_ids,anchors,keep) as capture:
            target(**example.inputs,use_cache=False,logits_to_keep=1)
            states=capture.states
            full_layer_inputs=full_input_capture.values
    comparisons=[]
    report=dict(passed=False,state_parity=comparisons,state_limits=STATE_LIMITS,reader_relative_l2_limit=READ_LIMIT)
    report['gate_scope']='same_prefix_only; cross_length_errors_retained_not_equivalence_claim'
    def persist():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    persist()
    for slot,a in enumerate(anchors[0].tolist()):
        inputs={k:(v[:,:a] if k in {'input_ids','attention_mask','mm_token_type_ids'} else v)
                for k,v in example.inputs.items()}
        with torch.no_grad(),RequestPositionState(target).activate():
            with GDNInputCapture(target,reader.layer_ids) as native_input_capture:
                output=target(**inputs,use_cache=True,output_hidden_states=True,logits_to_keep=1)
                native_layer_inputs=native_input_capture.values
        native=cache_reader_states(draft,output.past_key_values)
        entry=dict(anchor=a,layers=[])
        comparisons.append(entry)
        for layer,actual,reference in zip(reader.layer_ids,states,native):
            entry['layers'].append(dict(layer=layer,**state_pair_metrics(actual[:,slot],reference[:,0])))
        # Capture the SAME prefix independently to separate length-dependent
        # BF16 backbone differences from recurrence reconstruction differences.
        prefix_anchors=anchors.new_tensor([[a]])
        with torch.no_grad(),RequestPositionState(target).activate():
            with GDNInputCapture(target,reader.layer_ids) as prefix_input_capture, GDNAnchorCapture(target,reader.layer_ids,prefix_anchors,torch.ones_like(prefix_anchors,dtype=torch.bool)) as capture:
                target(**inputs,use_cache=False,logits_to_keep=1)
                prefix_states=capture.states
                prefix_layer_inputs=prefix_input_capture.values
        entry['layer_input_drift']=[dict(layer=layer,
            cross_length=input_pair_metrics(full_layer_inputs[layer][:,:a],native_layer_inputs[layer]),
            same_prefix=input_pair_metrics(prefix_layer_inputs[layer],native_layer_inputs[layer]))
            for layer in reader.layer_ids]
        entry['same_prefix_layers']=[dict(layer=layer,**state_pair_metrics(actual[:,0],reference[:,0]))
            for layer,actual,reference in zip(reader.layer_ids,prefix_states,native)]
        # Probe the real Reader input with frozen-backbone representations, not
        # its final output: up is zero at initialization and would hide errors.
        pending=output.logits[:,-1:].argmax(-1)
        from litespec.vlm.runtime import draft_block, selected_layers
        from litespec.vlm.context import ContextPolicy
        pivot=selected_layers(output.hidden_states,draft.target_layer_ids,draft.config.num_target_layers)[:,-1:].detach()
        hidden_at_reader=[]
        handle=reader.register_forward_pre_hook(lambda module,args:hidden_at_reader.append(args[0].detach()))
        try:
            with torch.no_grad():
                draft_block(draft,target.get_input_embeddings(),anchor_ids=pending,pivot=pivot,history=None,
                    anchor=a,prefill_length=example.prefill_length,vision_mask=example.vision_mask,
                    policy=ContextPolicy.no_history(),gdn_states=native)
        finally:handle.remove()
        if len(hidden_at_reader)!=1:raise RuntimeError('expected one Reader input capture')
        entry['reader_content']=read_pair_metrics(reader,hidden_at_reader[0],
            tuple(s[:,slot] for s in states),tuple(s[:,0] for s in native))
        entry['same_prefix_reader_content']=read_pair_metrics(reader,hidden_at_reader[0],
            tuple(s[:,0] for s in prefix_states),tuple(s[:,0] for s in native))
        passed=entry_gate(entry)
        persist()
        print(json.dumps(dict(state_comparison=entry),allow_nan=False),flush=True)
        if not passed:
            raise RuntimeError(f'GDN state/content mismatch; detailed metrics: {args.output}')
        del output,native
    online=aligned.CPUReferenceDLite(draft,target.get_output_embeddings(),target.get_input_embeddings(),
        draft.mask_token_id,draft.block_size,attention_backend='flex_attention',num_anchors=2,
        final_ce_weight=1.,l1_loss_weight=.1,base_lm_ce_weight=0.)
    # Independent SDPA blocks isolate Reader math; launcher smoke tests production Flex/FSDP.
    with RequestPositionState(target).activate():inputs=aligned.prepare_forward(target,online,example)
    with torch.no_grad():
        normal=online(**inputs).loss
        with reader_mode(reader,'off'):off=online(**inputs).loss
    torch.testing.assert_close(normal,off,rtol=0,atol=0)
    optimizer=torch.optim.AdamW(reader.parameters(),lr=.0001,weight_decay=0)
    history=[]
    for step in range(3):
        optimizer.zero_grad();loss=online(**inputs).loss;loss.backward()
        history.append(dict(step=step+1,loss=float(loss.detach()),
                            gradient_norms={n:float(p.grad.float().norm()) if p.grad is not None else None
                                            for n,p in reader.named_parameters()}))
        optimizer.step()
    if not history[-1]['gradient_norms']['query.weight'] or not history[-1]['gradient_norms']['value_down']:
        raise RuntimeError('Reader content projections have no gradient after warmup updates')
    if frozen_digest(draft.state_dict())!=cfg['frozen_backbone_sha256']:
        raise RuntimeError('frozen backbone changed')
    report.update(passed=True,zero_init_identity=True,frozen_backbone_unchanged=True,
                  reader_gradients=history,scope='single_GPU_SDPA_real_model_not_FSDP')
    persist()
    print(f'Passed real-model preflight: {args.output}',flush=True)


if __name__=='__main__':main()
