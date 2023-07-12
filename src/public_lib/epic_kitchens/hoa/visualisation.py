"""Tools for visualising hand-object detections"""
import os
import warnings
from copy import deepcopy
from typing import Tuple

import PIL.Image
from PIL import ImageFont, ImageDraw

from .types import FrameDetections, HandDetection, HandSide, HandState, ObjectDetection


class DetectionRenderer:
    """A class to render hand-object annotations onto the corresponding image"""
    def __init__(
        self,
        hand_threshold: float = 0.8,
        object_threshold: float = 0.01,
        only_interacted_objects: bool = True,
        font_size=20,
        border=4,
        text_padding=4,
    ):
        """

        Args:
            hand_threshold: Filter hand detections above this threshold
            object_threshold: Filter object detections above this threshold
            only_interacted_objects: Only draw objects that are part of an
                interaction with a hand
            font_size: The font-size for the bounding box labels
            border: The width of the border of annotation bounding boxes.
            text_padding: The amount of padding within bounding box annotation labels
        """
        self.hand_threshold = hand_threshold
        self.object_threshold = object_threshold
        self.only_interacted_objects = only_interacted_objects

        try:
            self.font = ImageFont.truetype(
                os.path.dirname(os.path.abspath(__file__)) + "/Roboto-Regular.ttf",
                size=font_size,
            )
        except IOError:
            warnings.warn(
                "Could not find font, falling back to Pillow default. "
                "`font_size` will not have an effect"
            )
            self.font = ImageFont.load_default()
        self.hand_rgb = {
            HandSide.LEFT.name: (0, 90, 181),
            HandSide.RIGHT.name: (220, 50, 32),
        }
        self.border = border
        self.text_padding = text_padding
        self.hand_rgba = {side: (*rgb, 70) for side, rgb in self.hand_rgb.items()}

        self.object_rgb = (255, 194, 10)
        self.object_rgba = (*self.object_rgb, 70)
        self._img: PIL.Image.Image
        self._detections: FrameDetections
        self._draw: ImageDraw.ImageDraw

        self.side2human = {HandSide.LEFT.name: "L", HandSide.RIGHT.name: "R"}
        self.state2human = {
            HandState.NO_CONTACT.name: "N",
            HandState.SELF_CONTACT.name: "S",
            HandState.ANOTHER_PERSON.name: "O",
            HandState.PORTABLE_OBJECT.name: "P",
            HandState.STATIONARY_OBJECT.name: "F",
        }

    def render_detections(
        self, frame: PIL.Image.Image, detections: FrameDetections, object_ids: int=-1, label_file_object_id: str=None, both_hand_objects_labels: bool=False, single_hand: str='both'
    ) -> PIL.Image.Image:
        """
        Args:
            frame: Frame to annotate with hand and object detections
            detections: Detections for the current frame
            object_ids: Ids detected for the object in the right hand
            label_file_object_id: path to the file containing Ids of objects

        Returns:
            A copy of ``frame`` annotated with the detections from ``detections``.
        """
        self._img = frame.copy()
        detections = self._detections = deepcopy(detections)
        detections.scale(
            width_factor=self._img.width, height_factor=self._img.height
        )
        
        hand_det_over_thresh = len([el for el in detections.hands if el.score >= self.hand_threshold])
        obj_det_over_thresh = len([el for el in detections.objects if el.score >= self.object_threshold])
        
        if len(detections.hands) == 0 and self.only_interacted_objects:
            return self._img
        
        if len(detections.objects) != 0:

            self._draw = ImageDraw.Draw(frame)
                        
            if not(hand_det_over_thresh == 0 or obj_det_over_thresh == 0):
                hand_object_idx_correspondences = detections.get_hand_object_interactions(
                    object_threshold=self.object_threshold, hand_threshold=self.hand_threshold
                )
            else:  
                hand_object_idx_correspondences = {}
            if not self.only_interacted_objects:
                for object in detections.objects:
                    if object.score >= self.object_threshold:
                        self._render_object(object)
                        
            
            if object_ids != -1:
                if both_hand_objects_labels:         
                    interacting = detections.get_hand_object_interactions(0, 0, one_hand_side=True)
                    labeled_object_right = [obj_idx for hand_idx, obj_idx in interacting.items() if detections.hands[hand_idx].side.value == HandSide.RIGHT.value]
                    labeled_object_left = [obj_idx for hand_idx, obj_idx in interacting.items() if detections.hands[hand_idx].side.value == HandSide.LEFT.value]
                    import json
                    with open(label_file_object_id, 'r') as f:
                        labels = json.load(f)
                else:    
                    interacting = detections.get_hand_object_interactions(0, 0, one_hand_side=True)
                    labeled_object = [obj_idx for hand_idx, obj_idx in interacting.items() if detections.hands[hand_idx].side.value == HandSide.RIGHT.value]
                    import json
                    with open(label_file_object_id, 'r') as f:
                        labels = json.load(f)
            

            for hand_idx, object_idx in hand_object_idx_correspondences.items():
                hand = detections.hands[hand_idx]
                if single_hand != 'both':
                    hand_side = HandSide.LEFT if single_hand == 'left' else HandSide.RIGHT
                    if hand.side.value != hand_side.value:
                        continue
                
                object = detections.objects[object_idx]
                if self.only_interacted_objects:
                    if object.score >= self.object_threshold:
                        label = None
                        if object_ids != -1:
                            if both_hand_objects_labels:
                                if len(labeled_object_right) > 0 and detections.objects[labeled_object_right[0]].bbox == object.bbox:
                                    label = str(labels[str(object_ids * 2)])
                                elif len(labeled_object_left) > 0 and detections.objects[labeled_object_left[0]].bbox == object.bbox:
                                    label = str(labels[str((object_ids * 2)+1)])
                                else:
                                    label = None
                            else:
                                if len(labeled_object) > 0 and detections.objects[labeled_object[0]].bbox == object.bbox:
                                    label = str(labels[str(object_ids)])
                                else:
                                    label = None
                        self._render_object(object, label)
                self._render_hand_object_correspondence(hand, object)
                
            for hand in detections.hands:
                if single_hand != 'both':
                    hand_side = HandSide.LEFT if single_hand == 'left' else HandSide.RIGHT
                    if hand.side.value != hand_side.value:
                        continue
                if hand.score >= self.hand_threshold:
                    self._render_hand(hand)
        else:
            for hand in detections.hands:
                if hand.score >= self.hand_threshold:
                    self._render_hand(hand)

        return self._img

    def _render_hand(self, hand: HandDetection):
        mask = PIL.Image.new("RGBA", self._img.size)
        mask_draw = ImageDraw.Draw(mask)
        hand_bbox = hand.bbox.coords_int
        color = self.hand_rgb[hand.side.name]
        mask_draw.rectangle(
            hand_bbox,
            outline=color,
            width=self.border,
            fill=self.hand_rgba[hand.side.name],
        )
        self._img.paste(mask, (0, 0), mask)
        self._render_label_box(
            ImageDraw.Draw(self._img),
            top_left=hand.bbox.top_left_int,
            text=f"{self.side2human[hand.side.name]}-{self.state2human[hand.state.name]}",
            padding=self.text_padding,
            outline_color=color,
        )

    def _render_object(self, object: ObjectDetection, label: str=None):
        mask = PIL.Image.new("RGBA", self._img.size)
        mask_draw = ImageDraw.Draw(mask)
        object_bbox = object.bbox.coords_int
        mask_draw.rectangle(
            object_bbox,
            outline=self.object_rgb,
            width=self.border,
            fill=self.object_rgba,
        )
        if label is not None:
            text_im = f"O: {label}"
        else:
            text_im = "O"
        self._img.paste(mask, (0, 0), mask)
        self._render_label_box(
            ImageDraw.Draw(self._img),
            top_left=object.bbox.top_left_int,
            text=text_im,
            padding=self.text_padding,
            outline_color=self.object_rgb,
        )

    def _render_hand_object_correspondence(
        self, hand: HandDetection, object: ObjectDetection
    ):
        hand_center = hand.bbox.center_int
        object_center = object.bbox.center_int
        draw = ImageDraw.Draw(self._img)
        draw.line(
            [hand_center, object_center],
            fill=self.hand_rgb[hand.side.name],
            width=self.border,
        )

        r = round(7 / 4 * self.border)

        x, y = hand_center
        draw.ellipse((x - r, y - r, x + r, y + r), fill=self.hand_rgb[hand.side.name])

        x, y = object_center
        draw.ellipse((x - r, y - r, x + r, y + r), fill=self.object_rgb)

    def _render_label_box(
        self,
        draw: ImageDraw.ImageDraw,
        top_left: Tuple[int, int],
        text: str,
        padding: int = 10,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        outline_color: Tuple[int, int, int] = (0, 0, 0),
        text_color: Tuple[int, int, int] = (0, 0, 0),
    ):
        text_size = draw.textsize(text, font=self.font)
        offset_x, offset_y = self.font.getoffset(text)
        text_width = text_size[0] + offset_x
        text_height = text_size[1] + offset_y
        x, y = top_left
        bottom_right = (
            x + self.border * 2 + padding * 2 + text_width,
            y + padding + text_height,
        )
        box_coords = [top_left, bottom_right]
        draw.rectangle(
            box_coords, fill=background_color, outline=outline_color, width=self.border,
        )
        text_coordinate = (
            x + self.border + padding - offset_x,
            y + self.border + padding - offset_y + 1,
        )
        draw.text(text_coordinate, text, font=self.font, fill=text_color)
