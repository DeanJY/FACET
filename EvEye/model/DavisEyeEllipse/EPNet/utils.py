import torch


class PredDecoder:
    def __init__(self, pred):
        self.pred = pred
        self.hm = self.pred["hm"].sigmoid_()
        self.ang = self.pred["ang"]
        self.ab = self.pred["ab"]
        self.reg = self.pred["reg"]

    def transpose_feat(self, feat):
        # feat.shape: (b, c, h, w)
        b, c, h, w = feat.size()
        # feat.shape: (b, c, h, w) -> (b, h, w, c)
        feat = feat.permute(0, 2, 3, 1)
        # feat.shape: (b, h, w, c) -> (b, h*w, c)
        feat = feat.view(b, h * w, c)
        feat = feat.contiguous()

        return feat

    def gather_feat(self, feat, ind, mask=None):
        # feat.shape: (b, h*w, c)
        feat_b, hw, c = feat.size()
        # ind.shape: (b, 100)
        ind_b, n = ind.size()
        assert feat_b == ind_b
        b = ind_b
        # ind.shape: (b, 100) -> (b, 100, 1) -> (b, 100, c)
        ind = ind.unsqueeze(2)
        ind = ind.expand(b, n, c)
        # feat.shape: (b, h*w, c) -> (b, 100, c)
        feat = feat.gather(1, ind)

        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, c)

        return feat

    def nms(self, heatmap, kernel=3):
        pad = (kernel - 1) // 2
        hmax = torch.nn.functional.max_pool2d(
            heatmap, (kernel, kernel), stride=1, padding=pad
        )
        keep = (hmax == heatmap).float()
        heatmap = heatmap * keep

        return heatmap

    def topk(self, heatmap, K=100):
        # heatmap.shape: (b, c, h, w)
        b, c, h, w = heatmap.size()
        # heatmap.shape: (b, c, h, w) -> (b, c, h*w)
        heatmap = heatmap.view(b, c, -1)

        # top_scores.shape: (b, c, K), top_inds.shape: (b, c, K)
        topk_scores, topk_inds = torch.topk(heatmap, K)
        # topk_inds.shape, topk_ys.shape, topk_xs.shape: (b, c, K)
        topk_inds = topk_inds % (h * w)
        topk_ys = (topk_inds / w).int().float()
        topk_xs = (topk_inds % w).int().float()

        # topk_scores.shape: (b, c, K) -> (b, c*K)
        topk_scores = topk_scores.view(b, -1)
        # topk_score.shape, topk_ind.shape: (b, c*K)
        topk_score, topk_ind = torch.topk(topk_scores, K)
        # topk_clses.shape: (b, c*K)
        topk_clses = (topk_ind / K).int()

        topk_inds = self.gather_feat(topk_inds.view(b, -1, 1), topk_ind).view(b, K)
        topk_ys = self.gather_feat(topk_ys.view(b, -1, 1), topk_ind).view(b, K)
        topk_xs = self.gather_feat(topk_xs.view(b, -1, 1), topk_ind).view(b, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def decode(self, K=100):
        b, c, h, w = self.hm.size()
        hm = self.nms(self.hm)
        scores, inds, clses, ys, xs = self.topk(hm)

        reg = self.transpose_feat(self.reg)
        reg = self.gather_feat(reg, inds)
        reg = reg.view(b, K, 2)

        xs = xs.view(b, K, 1) + reg[:, :, 0:1]
        ys = ys.view(b, K, 1) + reg[:, :, 1:2]

        ab = self.transpose_feat(self.ab)
        ab = self.gather_feat(ab, inds)
        ab = ab.view(b, K, 2)

        ang = self.transpose_feat(self.ang)
        ang = self.gather_feat(ang, inds)
        ang = ang.view(b, K, 1)

        clses = clses.view(b, K, 1).float()
        scores = scores.view(b, K, 1)
        bboxes = torch.cat([xs, ys, ab[..., 0:1], ab[..., 1:2], ang], dim=2)

        detections = torch.cat([bboxes, scores, clses], dim=2)

        return detections
