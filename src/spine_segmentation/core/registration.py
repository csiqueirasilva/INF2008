import torch
import torch.nn.functional as F

def sobel_kernels(dev):
    gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=dev)/8.0
    gy = gx.t()
    return gx.view(1,1,3,3), gy.view(1,1,3,3)

def edge_ncc_loss(pred, target, dev):
    # pred: [N,1,H,W] from DRR; target: [N,1,h,w] from query img
    if target.dtype != torch.float32:
        target = target.float()
    if pred.shape[-2:] != target.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)

    kx, ky = sobel_kernels(dev)
    px = F.conv2d(pred,   kx, padding=1)
    py = F.conv2d(pred,   ky, padding=1)
    tx = F.conv2d(target, kx, padding=1)
    ty = F.conv2d(target, ky, padding=1)

    pe = torch.sqrt(px*px + py*py + 1e-6)
    te = torch.sqrt(tx*tx + ty*ty + 1e-6)
    a = pe - pe.mean()
    b = te - te.mean()
    ncc = (a*b).mean() / (a.std()*b.std() + 1e-6)
    return 1.0 - ncc

def refine_pose(drr, qry_t, rot, trs, iters=250, lr=0.02, dev=None,
                patience=40, min_delta=1e-4):
    rot = torch.nn.Parameter(rot.clone())
    trs = torch.nn.Parameter(trs.clone())
    opt = torch.optim.Adam([rot, trs], lr=lr)
    best_loss, stalled = float("inf"), 0
    best = (rot.detach().clone(), trs.detach().clone())
    for _ in range(iters):
        opt.zero_grad()
        pred = drr(rot, trs, parameterization="euler_angles", convention="ZXY")
        loss = edge_ncc_loss(pred, qry_t, dev)
        loss.backward(); opt.step()
        val = float(loss.detach().cpu().item())
        if val + min_delta < best_loss:
            best_loss, stalled = val, 0
            best = (rot.detach().clone(), trs.detach().clone())
        else:
            stalled += 1
            if stalled >= patience:
                break
    return best + (best_loss,)
