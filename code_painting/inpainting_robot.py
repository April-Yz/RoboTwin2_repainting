# 反向选择忘记在inputvideo选择了，而是选择在targetvideo选
import torch
import numpy as np
import cv2
import glob
import torch.nn as nn
from typing import Any, Dict, List
from pathlib import Path
from PIL import Image
import os
import sys
import argparse
import tempfile
import imageio
import imageio.v2 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sam_segment import build_sam_model
from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from ostrack import build_ostrack_model, get_box_using_ostrack
from sttn_video_inpaint import build_sttn_model, \
    inpaint_video_with_builded_sttn
from pytracking.lib.test.evaluation.data import Sequence
from utils import dilate_mask, show_mask, show_points, get_clicked_point
import supervision as sv
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
current_dir = os.path.dirname(__file__)
grounded_sam2_dir = os.path.abspath(os.path.join(current_dir, "Grounded_SAM_2"))
sys.path.append(grounded_sam2_dir)
print(f"Grounded-SAM-2 dir: {grounded_sam2_dir}")
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from Grounded_SAM_2.utils.track_utils import sample_points_from_masks

GROUNDING_DINO_CONFIG   = f"{grounded_sam2_dir}/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPT  = f"{grounded_sam2_dir}/gdino_checkpoints/groundingdino_swint_ogc.pth"
SAM2_CFG_FILE           = f"configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPT            = f"{grounded_sam2_dir}/checkpoints/sam2.1_hiera_large.pt"
BOX_THRESHOLD, TEXT_THRESHOLD = 0.35, 0.25
PROMPT_TYPE_FOR_VIDEO   = "box"          # point / box / mask
TEXT_PROMPT             = "robotic arm and gripper."  # ↓你的目标描述


def setup_args(parser):
    parser.add_argument(
        "--input_video", type=str, required=True,
        help="Path to a single input video",
    )
    parser.add_argument(
        "--target_video", type=str, default=None,
        help="Path to a target video where masked content will be applied",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--invert_mask", action="store_true",
        help="Invert the mask to select the inverse region",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--tracker_ckpt", type=str, required=True,
        help="The path to tracker checkpoint.",
    )
    parser.add_argument(
        "--vi_ckpt", type=str, required=True,
        help="The path to video inpainter checkpoint.",
    )
    parser.add_argument(
        "--mask_idx", type=int, default=2, required=True,
        help="Which mask in the first frame to determine the inpaint region.",
    )
    parser.add_argument(
        "--fps", type=int, default=25, required=True,
        help="FPS of the input and output videos.",
    )

class RemoveAnythingVideo(nn.Module):
    def __init__(
            self, 
            args,
            tracker_target="ostrack",
            segmentor_target="sam",
            inpainter_target="sttn",
    ):
        super().__init__()
        tracker_build_args = {
            "tracker_param": args.tracker_ckpt
        }
        segmentor_build_args = {
            "model_type": args.sam_model_type,
            "ckpt_p": args.sam_ckpt
        }
        inpainter_build_args = {
            "lama": {
                "lama_config": args.lama_config,
                "lama_ckpt": args.lama_ckpt
            },
            "sttn": {
                "model_type": "sttn",
                "ckpt_p": args.vi_ckpt
            }
        }
        # --- build Grounding-DINO  -------------------------------------------------
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPT,
            device=self.device)
        # --- build SAM-2 image & video predictor -----------------------------------
        self.sam2_video_predictor = build_sam2_video_predictor(
            SAM2_CFG_FILE, SAM2_CHECKPT)
        sam2_img_model = build_sam2(SAM2_CFG_FILE, SAM2_CHECKPT)
        self.sam2_img_predictor = SAM2ImagePredictor(sam2_img_model)

        self.tracker = self.build_tracker(
            tracker_target, **tracker_build_args)
        self.segmentor = self.build_segmentor(
            segmentor_target, **segmentor_build_args)
        self.inpainter = self.build_inpainter(
            inpainter_target, **inpainter_build_args[inpainter_target])
        self.tracker_target = tracker_target
        self.segmentor_target = segmentor_target
        self.inpainter_target = inpainter_target

    def build_tracker(self, target, **kwargs):
        assert target == "ostrack", "Only support sam now."
        return build_ostrack_model(**kwargs)

    def build_segmentor(self, target="sam", **kwargs):
        assert target == "sam", "Only support sam now."
        return build_sam_model(**kwargs)

    def build_inpainter(self, target="sttn", **kwargs):
        if target == "lama":
            return build_lama_model(**kwargs)
        elif target == "sttn":
            return build_sttn_model(**kwargs)
        else:
            raise NotImplementedError("Only support lama and sttn")

    def forward_tracker(self, frames_ps, init_box):
        init_box = np.array(init_box).astype(np.float32).reshape(-1, 4)
        seq = Sequence("tmp", frames_ps, 'inpaint-anything', init_box)
        all_box_xywh = get_box_using_ostrack(self.tracker, seq)
        return all_box_xywh

    def forward_segmentor(self, img, point_coords=None, point_labels=None,
                          box=None, mask_input=None, multimask_output=True,
                          return_logits=False):
        self.segmentor.set_image(img)

        masks, scores, logits = self.segmentor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits
        )
        self.segmentor.reset_image()
        return masks, scores

    def forward_inpainter(self, frames, masks):
        if self.inpainter_target == "lama":
            frames_inpainted = frames.copy()
            for idx in range(len(frames)):
                frames_inpainted[idx] = inpaint_img_with_builded_lama(
                    self.inpainter, frames[idx], masks[idx], device=self.device)
        elif self.inpainter_target == "sttn":
            frames_pil = [Image.fromarray(frame) for frame in frames]
            masks_pil = [Image.fromarray(np.uint8(mask * 255)) for mask in masks]
            frames_inpainted = inpaint_video_with_builded_sttn(
                self.inpainter, frames_pil, masks_pil, device=self.device)
        else:
            raise NotImplementedError
        return frames_inpainted

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def mask_selection(self, masks, scores, ref_mask=None, interactive=False):
        if interactive:
            raise NotImplementedError
        else:
            if ref_mask is not None:
                mse = np.mean(
                    (masks.astype(np.int32) - ref_mask.astype(np.int32))**2,
                    axis=(-2, -1)
                )
                idx = mse.argmin()
            else:
                idx = scores.argmax()
            return masks[idx]

    @staticmethod
    def get_box_from_mask(mask):
        x, y, w, h = cv2.boundingRect(mask)
        return np.array([x, y, w, h])

    def forward(
        self,
        frame_ps: List[str],         # 已按顺序排好
        key_frame_idx: int,
        *_,
        # dilate_kernel_size: int = None,
        dilate_kernel_size: int = 13
    ):
        import tempfile, cv2, shutil    # 内联 import，避免全局污染
        from pathlib import Path
        assert key_frame_idx == 0, "当前版本只支持首帧交互"

        # ---------- Step-0: 读首帧 ------------------------------------------------
        first_frame_p = frame_ps[0]
        output_path = '/home/pine/RoboTwin2/third_party/Inpaint-Anything/modified_frame1.png'
        frame_img = cv2.imread(first_frame_p)  # 读取图像
        cv2.imwrite(output_path, frame_img)
        image_source, image_dino = load_image(first_frame_p)    # dino RGB
        h, w, _ = image_source.shape
        

        # ---------- Step-1: GDINO 检测框 -----------------------------------------
        print("Grounding-DINO 正在检测 ...")
        
        boxes, confs, labels = predict(
            model=self.grounding_model,
            image=image_dino,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        input_boxes_xyxy = (
            box_convert(boxes, "cxcywh", "xyxy")
            .cpu().numpy()
            .copy()                                 # 避免“not writable”警告
        )

        if input_boxes_xyxy.shape[0] == 0:
            raise RuntimeError(
                "Grounding-DINO 未检测到任何框，请检查 TEXT_PROMPT 或阈值。"
            )

        # ---------- Step-2: SAM-2 (image) 获取初始 mask ---------------------------
        self.sam2_img_predictor.set_image(image_source)

        masks_list, scores_list = [], []
        for box in input_boxes_xyxy.astype(np.float32):         # (4,)
            m, s, _ = self.sam2_img_predictor.predict(
                box=box[None, :], multimask_output=False
            )
            if m is None or m.size == 0:                # 跳过无效 mask
                continue
            masks_list.append(m.squeeze())              # (H, W)
            scores_list.append(s.item())

        if not masks_list:
            raise RuntimeError(
                "SAM-2 未能为任何框生成有效 mask，请检查框尺寸或图片分辨率。"
            )

        masks  = np.stack(masks_list, axis=0)           # (N, H, W)
        scores = np.array(scores_list)                  # (N,)

        # ---------- Step-2.5: 生成连续 JPG 临时目录 -------------------------------
        tmp_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
        for idx, p in enumerate(frame_ps):
            img_rgb = iio.imread(p)
            cv2.imwrite(str(tmp_dir / f"{idx:05d}.jpg"),
                        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # ---------- Step-3: 初始化视频 predictor ---------------------------------
        state = self.sam2_video_predictor.init_state(video_path=str(tmp_dir))

        if PROMPT_TYPE_FOR_VIDEO == "box":
            for obj_id, box in enumerate(input_boxes_xyxy, start=1):
                self.sam2_video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0, obj_id=obj_id, box=box)
        elif PROMPT_TYPE_FOR_VIDEO == "mask":
            for obj_id, mask in enumerate(masks, start=1):
                self.sam2_video_predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0, obj_id=obj_id, mask=mask)
        else:  # point
            pts_all = sample_points_from_masks(masks, num_points=10)
            for obj_id, pts in enumerate(pts_all, start=1):
                labels = np.ones(len(pts), dtype=np.int32)
                self.sam2_video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0, obj_id=obj_id,
                    points=pts, labels=labels)

        # ---------- Step-4: Propagate 全视频 mask ---------------------------------
        video_segments = {}
        for f_idx, obj_ids, mask_logits in self.sam2_video_predictor.propagate_in_video(state):
            video_segments[f_idx] = {
                obj_id: (mask_logits[i] > 0).cpu().numpy()
                for i, obj_id in enumerate(obj_ids)
            }

        # ---------- Step-5: 整理输出格式 ------------------------------------------
        # all_mask, all_box, all_frame = [], [], []
        # for i, frame_p in enumerate(frame_ps):
        #     frame = iio.imread(frame_p)
        #     masks_i = video_segments.get(i, {})
        #     if len(masks_i) == 0:    # 无目标
        #         mask_union = np.zeros(frame.shape[:2], dtype=np.uint8)
        #     else:
        #         mask_union = np.any(list(masks_i.values()), axis=0)
        #     if dilate_kernel_size:
        #         mask_union = dilate_mask(mask_union, dilate_kernel_size)

        #     # 计算 XYWH bbox（若无目标返回零框）
        #     if np.any(mask_union):
        #         x, y, w_, h_ = self.get_box_from_mask(mask_union.astype(np.uint8))
        #         box_xywh = np.array([x, y, w_, h_])
        #     else:
        #         box_xywh = np.zeros(4, dtype=np.int32)

        #     all_frame.append(frame)
        #     all_mask.append(mask_union.astype(np.uint8))
        #     all_box.append(box_xywh)
        print("Segmenting ...")
        all_mask, all_box, all_frame = [], [], []
        for i, frame_p in enumerate(frame_ps):
            frame = iio.imread(frame_p)
            segs  = video_segments.get(i, {})          # {obj_id: (H,W) or (1,H,W)}
            if segs:
                # 把每个 seg squeeze 成 (H,W) 再求并集
                masks_i = [np.squeeze(m.astype(bool)) for m in segs.values()]
                mask_union = np.any(masks_i, axis=0)   # (H, W)  bool
            else:
                mask_union = np.zeros(frame.shape[:2], dtype=bool)

            if dilate_kernel_size: # 可选膨胀
                # breakpoint()
                mask_union = dilate_mask(mask_union.astype(np.uint8),
                                        dilate_kernel_size).astype(bool)

            # --------- 保证 mask 最终就是 (H, W) ----------
            mask_union = np.squeeze(mask_union)
            assert mask_union.ndim == 2, f"mask ndim={mask_union.ndim}"

            # bbox
            if mask_union.any():
                x, y, w_, h_ = self.get_box_from_mask(mask_union.astype(np.uint8))
                box_xywh = np.array([x, y, w_, h_])
            else:
                box_xywh = np.zeros(4, dtype=np.int32)

            all_frame.append(frame)
            all_mask.append(mask_union.astype(np.uint8))   # ← 只保存 2-D uint8
            all_box.append(box_xywh)
            
        # breakpoint()  # 调试用，查看中间结果
            

        # ---------- Step-6: Inpaint ------------------------------------------------
        print("Inpainting ...")
        inpainted_frames = self.forward_inpainter(all_frame, all_mask)
        return inpainted_frames, all_mask, all_box
    
    # def forward(
    #         self,
    #         frame_ps: List[str],
    #         key_frame_idx: int,
    #         key_frame_point_coords: np.ndarray,
    #         key_frame_point_labels: np.ndarray,
    #         key_frame_mask_idx: int = None,
    #         dilate_kernel_size: int = 15,
    # ):
    #     """
    #     Mask is 0-1 ndarray in default
    #     Frame is 0-255 ndarray in default
    #     """
    #     assert key_frame_idx == 0, "Only support key frame at the beginning."

    #     # get key-frame mask
    #     key_frame_p = frame_ps[key_frame_idx]
    #     key_frame = iio.imread(key_frame_p)
    #     key_masks, key_scores = self.forward_segmentor(
    #         key_frame, key_frame_point_coords, key_frame_point_labels)

    #     # key-frame mask selection
    #     if key_frame_mask_idx is not None:
    #         key_mask = key_masks[key_frame_mask_idx]
    #     else:
    #         key_mask = self.mask_selection(key_masks, key_scores)
        
    #     if dilate_kernel_size is not None:
    #         key_mask = dilate_mask(key_mask, dilate_kernel_size)

    #     # get key-frame box
    #     key_box = self.get_box_from_mask(key_mask)

    #     # get all-frame boxes using video tracker
    #     print("Tracking ...")
    #     all_box = self.forward_tracker(frame_ps, key_box)

    #     # get all-frame masks using sam
    #     print("Segmenting ...")
    #     all_mask = [key_mask]
    #     all_frame = [key_frame]
    #     ref_mask = key_mask
    #     for frame_p, box in zip(frame_ps[1:], all_box[1:]):
    #         frame = iio.imread(frame_p)

    #         # XYWH -> XYXY
    #         x, y, w, h = box
    #         sam_box = np.array([x, y, x + w, y + h])
    #         masks, scores = self.forward_segmentor(frame, box=sam_box)
    #         # mask = self.mask_selection(masks, scores, ref_mask)
    #         mask = self.mask_selection(masks, scores)
    #         if dilate_kernel_size is not None:
    #             mask = dilate_mask(mask, dilate_kernel_size)

    #         ref_mask = mask
    #         all_mask.append(mask)
    #         all_frame.append(frame)

    #     # get all-frame inpainted results
    #     print("Inpainting ...")
    #     all_frame = self.forward_inpainter(all_frame, all_mask)
    #     return all_frame, all_mask, all_box


def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)

def show_img_with_mask(img, mask):
    if np.max(mask) == 1:
        mask = np.uint8(mask * 255)
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    plt.axis('off')
    show_mask(plt.gca(), mask, random_color=False)
    tmp_p = mkstemp(".png")
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

def show_img_with_point(img, point_coords, point_labels):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    plt.imshow(img)
    plt.axis('off')
    show_points(plt.gca(), point_coords, point_labels,
                size=(width * 0.04) ** 2)
    tmp_p = mkstemp(".png")
    plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)

def show_img_with_box(img, box):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    fig, ax = plt.subplots(1, figsize=(width / dpi / 0.77, height / dpi / 0.77))
    ax.imshow(img)
    ax.axis('off')

    x1, y1, w, h = box
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                             edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    tmp_p = mkstemp(".png")
    fig.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
    plt.close()
    return iio.imread(tmp_p)



if __name__ == "__main__":
    """Example usage:
    python remove_anything_video.py \
        --input_video ./example/video/paragliding/original_video.mp4 \
        --coords_type key_in \
        --point_coords 652 162 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt ./pretrained_models/big-lama \
        --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
        --vi_ckpt ./pretrained_models/sttn.pth \
        --mask_idx 2 \
        --fps 25
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import logging
    logger = logging.getLogger('imageio')
    logger.setLevel(logging.ERROR)

    dilate_kernel_size = args.dilate_kernel_size
    key_frame_mask_idx = args.mask_idx
    video_raw_p = args.input_video
    target_video_p = args.target_video
    video_name = Path(video_raw_p).stem
    # breakpoint()
    frame_raw_glob = None
    fps = args.fps
    num_frames = 10000
    output_dir = args.output_dir
    output_dir = Path(f"{output_dir}")
    frame_mask_dir = output_dir / f"mask_{video_name}"
    video_mask_p = output_dir / f"mask_{video_name}.mp4"
    video_rm_w_mask_p = output_dir / f"removed_w_mask_{video_name}.mp4"
    video_w_mask_p = output_dir / f"w_mask_{video_name}.mp4"
    video_w_box_p = output_dir / f"w_box_{video_name}.mp4"
    video_target_masked_p = output_dir / f"target_masked_{video_name}.mp4"
    frame_mask_dir.mkdir(exist_ok=True, parents=True)

    # load raw video or raw frames
    if Path(video_raw_p).exists():
        # all_frame = iio.mimread(video_raw_p)
        all_frame = iio.mimread(video_raw_p, memtest=False)
        fps = imageio.v3.immeta(video_raw_p, exclude_applied=False)["fps"]

        # tmp frames
        frame_ps = []
        for i in range(len(all_frame)):
            frame_p = str(mkstemp(suffix=f"{i:0>6}.png"))
            frame_ps.append(frame_p)
            iio.imwrite(frame_ps[i], all_frame[i])
            
    # Load target video if provided
    target_frames = None
    if target_video_p and Path(target_video_p).exists():
        target_frames = iio.mimread(target_video_p, memtest=False)
        target_fps = imageio.v3.immeta(target_video_p, exclude_applied=False)["fps"]
        # Ensure consistent frame rate
        if abs(target_fps - fps) > 1:
            print(f"Warning: Target video fps ({target_fps}) differs from source ({fps})")
    else:
        assert frame_raw_glob is not None
        frame_ps = sorted(glob.glob(frame_raw_glob))
        all_frame = [iio.imread(frame_p) for frame_p in frame_ps]
        fps = 25
        # save tmp video
        iio.mimwrite(video_raw_p, all_frame, fps=fps)

    num_frames = len(all_frame)
    num_frames = 300
    frame_start = 0

    frame_ps = frame_ps[frame_start:num_frames]
    all_frame = all_frame[frame_start:num_frames]
    
    point_labels = np.array(args.point_labels)
    if args.coords_type == "click":
        point_coords = get_clicked_point(frame_ps[0])
    elif args.coords_type == "key_in":
        point_coords = args.point_coords
    point_coords = np.array([point_coords])

    # inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RemoveAnythingVideo(args)
    model.to(device)
    with torch.no_grad():
        all_frame_rm_w_mask, all_mask, all_box = model(
            frame_ps, 0, point_coords, point_labels, key_frame_mask_idx,
            dilate_kernel_size
        )
    # Process inverted mask if needed
    if args.invert_mask:
        print("Inverting masks...")
        all_mask = [1 - mask for mask in all_mask]  # Invert mask (0->1, 1->0)
    
    # Visual removed results
    iio.mimwrite(video_rm_w_mask_p, all_frame_rm_w_mask, fps=fps)

    # Visual mask
    print("Saving masks ...")
    all_mask_uint8 = [np.uint8(mask * 255) for mask in all_mask]
    for i in range(len(all_mask_uint8)):
        mask_p = frame_mask_dir / f"{i:0>6}.jpg"
        iio.imwrite(mask_p, all_mask_uint8[i])
    iio.mimwrite(video_mask_p, all_mask_uint8, fps=fps)
    
    # Visual video with mask
    print("Saving video with mask ...")
    tmp = []
    for i in range(len(all_mask)):
        tmp.append(show_img_with_mask(all_frame[i], all_mask[i]))
    iio.mimwrite(video_w_mask_p, tmp, fps=fps)
    
    tmp = []
    # Visual video with box
    print("Saving video with box ...")
    for i in range(len(all_box)):
        tmp.append(show_img_with_box(all_frame[i], all_box[i]))
    iio.mimwrite(video_w_box_p, tmp, fps=fps)
    
    # Apply mask to target video if available
    if target_frames is not None:
        print("Applying masks to target video...")
        # Ensure frame count matches
        min_frames = min(len(target_frames), len(all_mask))
        target_masked_frames = []
        
        for i in range(min_frames):
            # Apply mask to target video frame
            # 调整mask尺寸以匹配目标视频帧
            target_h, target_w = target_frames[i].shape[:2]
            mask_resized = cv2.resize(all_mask[i], (target_w, target_h), 
                                      interpolation=cv2.INTER_NEAREST)
            # Create a 3-channel mask
            mask_3ch = np.stack([mask_resized] * 3, axis=2)
            # Apply mask to target frame
            masked_frame = target_frames[i] * mask_3ch
            target_masked_frames.append(masked_frame.astype(np.uint8))
        
        # Save masked target video
        iio.mimwrite(video_target_masked_p, target_masked_frames, fps=fps)
        print(f"Target masked video saved to {video_target_masked_p}")
