"""Bounded BF16 state diagnostics; never treat sparse allclose failures alone as correctness."""
import torch
import torch.nn.functional as F


STATE_LIMITS = dict(relative_l2=.005, worst_head_relative_l2=.02,
                    old_tolerance_outlier_fraction=.0001, max_old_tolerance_units=4.)
READ_LIMIT = .01


def state_pair_metrics(actual, reference):
    if actual.shape != reference.shape or actual.ndim != 4:
        raise ValueError('expected matching [B,H,K,V] state matrices')
    a,r=actual.detach().float(),reference.detach().float()
    if not bool(torch.isfinite(a).all() and torch.isfinite(r).all()):
        return dict(passed=False,finite=False)
    difference=a-r
    error=difference.abs()
    tolerance=.003+.02*r.abs()
    old_units=error/tolerance
    result=dict(finite=True,max_abs_error=float(error.max()),
        relative_l2=float(difference.norm()/r.norm().clamp_min(1e-12)),
        worst_head_relative_l2=float((difference.norm(dim=(-2,-1))/r.norm(dim=(-2,-1)).clamp_min(1e-12)).max()),
        old_tolerance_outlier_count=int((old_units>1).sum()),
        old_tolerance_outlier_fraction=float((old_units>1).float().mean()),
        max_old_tolerance_units=float(old_units.max()),
        strict_allclose_passed=bool((old_units<=1).all()),elements=r.numel())
    result['passed']=all(result[key]<=limit for key,limit in STATE_LIMITS.items())
    return result


@torch.no_grad()
def reader_content(reader, hidden, states):
    """Actual low-rank content path BEFORE log-RMS concatenation and zero-initialized up."""
    b,n,d=hidden.shape
    query=reader.query(F.rms_norm(hidden.float(),(d,)).to(hidden.dtype))
    query=query.reshape(b,n,len(states),reader.heads,reader.rank)
    keys=torch.einsum('bnlhr,lhrk->bnlhk',query,reader.query_up)
    reads=[]
    for i,state in enumerate(states):
        state=state.float()
        rms=(state.square().mean((-2,-1),keepdim=True)+1e-6).sqrt()
        normalized=(state/rms).transpose(-1,-2).to(hidden.dtype)
        compressed=torch.einsum('hvr,bhvk->bhrk',reader.value_down[i],normalized)
        reads.append(torch.einsum('bhrk,bnhk->bnhr',compressed,keys[:,:,i])/(reader.dk**.5))
    return torch.stack(reads,2).float()


@torch.no_grad()
def read_pair_metrics(reader, hidden, actual, reference):
    a=reader_content(reader,hidden,actual)
    r=reader_content(reader,hidden,reference)
    if not bool(torch.isfinite(a).all() and torch.isfinite(r).all()):
        return dict(passed=False,finite=False)
    relative=float((a-r).norm()/r.norm().clamp_min(1e-12))
    return dict(passed=relative<=READ_LIMIT,finite=True,relative_l2=relative,
                max_abs_error=float((a-r).abs().max()),reference_rms=float(r.square().mean().sqrt()),
                scope='initialized_Reader_content_before_zero_up_not_trained_Reader_equivalence')
