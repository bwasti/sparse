import numpy as np
import scipy.sparse as sparse
import torch
from torch.autograd import Function

## Naming
# cs, input channel index
# ks, output channel index
# vs, values at index (vs is overloaded to map a zorder, because "values" are just 1)
# ob_size, output block size (not the same as len(ks), because sparsity)
# ib_size, input block size
# idx, iterator over ks (passed as arg for transposed case)
# layout, binary matrix representing block sparsity


def zorder_vs(cs, ks, vs, disable=False):
    if disable:
        return list(zip(cs, ks))
    updat_list = list()

    def z_order_2d(x, y):
        answer = 0
        bits = max(len(bin(x)), len(bin(y))) - 2
        for i in range(bits):
            mshifted = 1 << i
            shift = i
            answer |= ((x & mshifted) << shift) | ((y & mshifted) << (shift + 1))
        return answer

    blk = 0
    for _, i in sorted([(z_order_2d(cs[i], ks[i]), i) for i in range(len(vs))]):
        vs[i] = blk
        updat_list.append((cs[i], ks[i]))
        blk += 1
    return updat_list


def create_sparse_ckv(layout):
    csr = sparse.csr_matrix(layout)
    cs, ks, vs = sparse.find(csr)
    updat_list = zorder_vs(cs, ks, vs)
    return cs, ks, vs, np.array(updat_list, dtype=np.int32)


# create lookup table
def create_seg_list(ob_size, cs, ks, vs, idx, max_seg=8, min_seg=2):
    channels = [0 for k in range(ob_size)]
    for i in idx:
        channels[ks[i]] += 1

    seg_count = 0
    seg = list()
    segs = list()
    kset = set()
    locks = 0
    lockids = dict()

    K = ks[idx[0]]
    for i in idx:
        c, k, v = cs[i], ks[i], vs[i]
        kset.add(k)
        # check for new value of k
        if k != K:
            if len(seg):
                segs.append((K, seg))
                seg = list()
                seg_count += 1
            # for more than one segment we need to use spin locks to sync accumulation
            if seg_count > 1:
                locks += 1
                lockids[K] = locks
            seg_count = 0
            K = k

        seg.append((c, v))
        channels[k] -= 1

        if len(seg) >= max_seg and channels[k] >= min_seg:
            segs.append((k, seg))
            seg = list()
            seg_count += 1

    # last k
    if len(seg):
        segs.append((k, seg))
        seg_count += 1
        if seg_count > 1:
            locks += 1
            lockids[k] = locks

    # handle empty ks
    for k in range(ob_size):
        if k not in kset:
            segs.append((k, []))

    return segs, lockids


def create_lut(segs, lockids, blocks):
    # now we create a lookup table
    header_offset = len(segs) * 4
    lut = np.empty(header_offset + blocks * 2, dtype=np.int32)
    # keep track of shared memory
    lut_max = 0
    offset = header_offset
    for i, (k, seg) in enumerate(segs):
        # this is a lookup table header: int2 offset to the entries, segment_size, k, lock_id
        lut[i * 4 : (i + 1) * 4] = offset // 2, len(seg), k, lockids.get(k, 0)
        lut_max = max(lut_max, len(seg))
        for entry in seg:  # entry is c, v
            lut[offset : offset + 2] = entry
            offset += 2
    return lut


def compress_w(W, layout, block_size, z_map):
    ib_size = W.shape[0] // block_size
    ob_size = W.shape[1] // block_size
    W_out = [np.array([]) for _ in range(len(z_map))]
    for c in range(ib_size):
        for k in range(ob_size):
            if layout[c][k]:
                block = W[
                    c * block_size : (c + 1) * block_size,
                    k * block_size : (k + 1) * block_size,
                ]
                W_out[z_map[(k, c)]] = block.cpu().numpy()
    return np.array(W_out)


class BlockSparseTensor:
    def __init__(self, W, layout, max_seg=1024, min_seg=0, device=torch.device("cuda")):
        assert layout.sum() > 0

        # track original dimensions
        self.shape = W.shape
        assert len(W.shape) == len(layout.shape)

        # save block size
        self.block_size = W.shape[0] // layout.shape[0]
        assert self.block_size <= 128
        assert W.shape[0] % layout.shape[0] == 0
        assert W.shape[1] % layout.shape[1] == 0
        assert W.shape[1] // layout.shape[1] == self.block_size

        cs, ks, vs, updat_lut = create_sparse_ckv(layout)
        idx = range(len(vs))
        idxT = sorted(idx, key=lambda i: cs[i])
        zmap = dict()
        for c, k, v in zip(cs, ks, vs):
            zmap[(int(k), int(c))] = int(v)
        num_blocks = len(vs)

        # actual tensor data
        self.data = torch.tensor(
            compress_w(W.detach(), layout, self.block_size, zmap),
            dtype=torch.float,
            device=device,
        )
        self.data.requires_grad = W.requires_grad

        # forward prop lookup table
        fsegs, f_lockids = create_seg_list(
            layout.shape[1], cs, ks, vs, idx, max_seg, min_seg
        )
        max_fseg_len = max([len(fseg[1]) for fseg in fsegs])
        flut = create_lut(fsegs, f_lockids, num_blocks)

        self.num_fsegs = len(fsegs)
        self.max_fseg = max_fseg_len
        self.flut = torch.tensor(flut, dtype=torch.int32, device=device)

        # backprop (dX) lookup table
        bsegs, b_lockids = create_seg_list(
            layout.shape[0], ks, cs, vs, idxT, max_seg, min_seg
        )
        max_bseg_len = max([len(bseg[1]) for bseg in bsegs])
        blut = create_lut(bsegs, b_lockids, num_blocks)

        self.num_bsegs = len(bsegs)
        self.max_bseg = max_bseg_len
        self.blut = torch.tensor(blut, dtype=torch.int32, device=device)

        # backprop/update (dW) lookup table
        self.ulut = torch.tensor(updat_lut, dtype=torch.int32, device=device)


def mm_raw(X, W: BlockSparseTensor):
    return torch.ops.sparse.bsmm_raw(
        X, W.data, W.shape[1], W.flut, W.block_size, W.num_fsegs, W.max_fseg
    )


def mm(X, W: BlockSparseTensor, transb=False):
    return torch.ops.sparse.bsmm(
        X,
        W.data,
        W.flut,
        W.blut,
        W.ulut,
        W.shape[1],
        W.block_size,
        W.num_fsegs,
        W.max_fseg,
        W.num_bsegs,
        W.max_bseg,
        transb,
    )


def mm_out(X, Y, W: BlockSparseTensor):
    W.data = torch.ops.sparse.mmbs(
        X.t(),
        Y,
        W.ulut,
        W.flut,
        W.blut,
        W.block_size,
        W.num_fsegs,
        W.max_fseg,
        W.num_bsegs,
        W.max_bseg,
    )
    return W


from torch.nn import init
import math

cuda = torch.device("cuda")


class SparseLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, sparsity_or_layout, block_size):
        super(SparseLinear, self).__init__()
        ib = in_features // block_size
        ob = out_features // block_size
        if type(sparsity_or_layout) is float:
            sparsity = sparsity_or_layout
            numel = ib * ob
            layout = np.zeros(numel, dtype=np.int32)
            layout[0 : int(numel * (1 - sparsity))] = 1
            np.random.shuffle(layout)
            layout = layout.reshape(ib, ob)
        else:
            layout = sparsity_or_layout
            sparsity = layout.sum() / layout.size

        self.in_features = in_features
        self.out_features = out_features
        data = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        init.kaiming_uniform_(data, a=math.sqrt(5))
        self.W = BlockSparseTensor(data, layout)

    def forward(self, input):
        return mm(input, self.W)

    def reset_parameters(self):
        data = torch.nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        init.kaiming_uniform_(data, a=math.sqrt(5))
        self.W = BlockSparseTensor(data, layout)
