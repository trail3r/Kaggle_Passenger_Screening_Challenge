import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import roi


def empty_report():
    return {
        "detected_views": 0,
        "lower_body_views": [],
        "rotation_candidates": [],
        "rotation_views": [],
        "structural_views": [],
        "arm_views": [],
        "rejected_arm_views": [],
        "interpolated_views": [],
        "rotation_error_before": None,
        "rotation_p90_before": None,
        "rotation_error_after": None,
        "rotation_p90_after": None,
        "mask_inclusion_before": 0.0,
        "mask_inclusion_after": 0.0,
        "anchor_violations": 0,
        "valid_keypoints": 0,
        "valid_rois": 0,
        "pose_rois": 0,
        "relative_rois": 0,
        "mean_roi_quality": 0.0,
    }


def empty_checkpoint(scan_id="scan_a", signature=None, model_hash="b" * 64, source_hash=None):
    source_hash = source_hash or roi.file_sha256(Path(roi.__file__))
    signature = signature or roi.pipeline_signature(model_hash, source_hash)
    artifact = {
        "schema_version": np.array(roi.SCHEMA_VERSION, dtype=np.int16),
        "artifact_version": np.array(roi.ARTIFACT_VERSION),
        "pipeline_signature": np.array(signature),
        "model_sha256": np.array(model_hash),
        "source_sha256": np.array(source_hash),
        "scan_id": np.array(scan_id),
        "pose_report": np.array(json.dumps(empty_report())),
    }

    for key, shape in roi.CHECKPOINT_SHAPES.items():
        dtype = roi.CHECKPOINT_DTYPES[key]

        if key in {"polygons", "boxes", "body_boxes"}:
            value = np.full(shape, -1, dtype=dtype)
        elif key == "rotation_prediction":
            value = np.full(shape, np.nan, dtype=dtype)
        elif key == "origin_joint":
            value = np.full(shape, -1, dtype=dtype)
        else:
            value = np.zeros(shape, dtype=dtype)

        artifact[key] = value

    return artifact


def aggregate_from_checkpoints(*artifacts):
    return {
        "schema_version": np.array(roi.SCHEMA_VERSION, dtype=np.int16),
        "artifact_version": np.array(roi.ARTIFACT_VERSION),
        "pipeline_signature": artifacts[0]["pipeline_signature"].copy(),
        "model_sha256": artifacts[0]["model_sha256"].copy(),
        "source_sha256": artifacts[0]["source_sha256"].copy(),
        "scan_ids": np.asarray([str(artifact["scan_id"]) for artifact in artifacts]),
        **{key: np.stack([artifact[key] for artifact in artifacts]) for key in roi.AGGREGATE_KEYS},
    }


def clean_rotation_pose():
    keypoints = np.zeros((roi.VIEW, roi.KEYPOINT_COUNT, 2), dtype=np.float32)
    confidence = np.zeros((roi.VIEW, roi.KEYPOINT_COUNT), dtype=np.float32)
    center = 256
    pairs = [
        (roi.LEFT_SHOULDER, roi.RIGHT_SHOULDER, 75, 185),
        (roi.LEFT_ELBOW, roi.RIGHT_ELBOW, 105, 135),
        (roi.LEFT_WRIST, roi.RIGHT_WRIST, 125, 85),
        (roi.LEFT_HIP, roi.RIGHT_HIP, 48, 355),
        (roi.LEFT_KNEE, roi.RIGHT_KNEE, 38, 490),
        (roi.LEFT_ANKLE, roi.RIGHT_ANKLE, 32, 625),
    ]

    for view, angle in enumerate(roi.VIEW_ANGLES):
        for left, right, amplitude, y in pairs:
            keypoints[view, left] = [center + amplitude * np.cos(angle), y]
            keypoints[view, right] = [center - amplitude * np.cos(angle), y]
            confidence[view, [left, right]] = 0.9

    return keypoints, confidence


class ArtifactValidationTest(unittest.TestCase):
    def test_empty_checkpoint_and_aggregate_are_valid(self):
        checkpoint = empty_checkpoint()
        self.assertEqual(roi.validate_checkpoint(checkpoint, "scan_a", str(checkpoint["pipeline_signature"])), [])
        aggregate = aggregate_from_checkpoints(checkpoint)
        self.assertEqual(roi.validate_aggregate(aggregate, ["scan_a"]), [])

    def test_keypoint_metadata_must_agree(self):
        checkpoint = empty_checkpoint()
        checkpoint["keypoints"][0, roi.LEFT_SHOULDER] = [100, 100]
        checkpoint["keypoint_confidence"][0, roi.LEFT_SHOULDER] = 0.9
        checkpoint["keypoint_source"][0, roi.LEFT_SHOULDER] = roi.KEYPOINT_OBSERVED
        checkpoint["origin_joint"][0, roi.LEFT_SHOULDER] = roi.LEFT_SHOULDER

        errors = roi.validate_checkpoint(checkpoint)
        self.assertTrue(any("validity" in error for error in errors))

        checkpoint["keypoint_valid"][0, roi.LEFT_SHOULDER] = True
        checkpoint["origin_joint"][0, roi.LEFT_SHOULDER] = -1
        errors = roi.validate_checkpoint(checkpoint)
        self.assertTrue(any("observed keypoint" in error for error in errors))

    def test_rotation_prediction_is_jointwise_finite_or_nan(self):
        checkpoint = empty_checkpoint()
        checkpoint["rotation_prediction"][:, roi.LEFT_HIP] = 10
        self.assertEqual(roi.validate_checkpoint(checkpoint), [])

        checkpoint["rotation_prediction"][0, roi.LEFT_HIP, 0] = np.nan
        errors = roi.validate_checkpoint(checkpoint)
        self.assertTrue(any("rotation prediction" in error for error in errors))

    def test_roi_polygon_and_box_must_agree(self):
        checkpoint = empty_checkpoint()
        polygon = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)
        checkpoint["polygons"][0, 0] = polygon
        checkpoint["boxes"][0, 0] = roi.polygon_box(polygon)
        checkpoint["roi_valid"][0, 0] = True
        checkpoint["roi_visible"][0, 0] = True
        checkpoint["roi_visibility"][0, 0] = 1
        checkpoint["roi_reliable"][0, 0] = True
        checkpoint["roi_quality"][0, 0] = 0.5
        checkpoint["roi_source"][0, 0] = roi.ROI_ORIENTED
        report = empty_report()
        report.update({"valid_rois": 1, "pose_rois": 1, "mean_roi_quality": 0.5})
        checkpoint["pose_report"] = np.array(json.dumps(report))
        self.assertEqual(roi.validate_checkpoint(checkpoint), [])

        checkpoint["boxes"][0, 0] = [100, 100, 120, 120]
        errors = roi.validate_checkpoint(checkpoint)
        self.assertTrue(any("does not match polygon" in error for error in errors))

        checkpoint["boxes"][0, 0] = roi.polygon_box(polygon)
        checkpoint["polygons"][0, 0] = np.array([[10, 10]] * 4, dtype=np.float32)
        errors = roi.validate_checkpoint(checkpoint)
        self.assertTrue(any("invalid geometry" in error for error in errors))


class PoseRefinementTest(unittest.TestCase):
    def test_whole_view_swap_is_corrected(self):
        clean, confidence = clean_rotation_pose()
        corrupted = clean.copy()
        view = 2

        for left, right in roi.SYMMETRIC_JOINTS:
            corrupted[view, [left, right]] = corrupted[view, [right, left]]

        frame_confidence = np.ones(roi.VIEW, dtype=np.float32)
        masks = np.ones((roi.VIEW, roi.HEIGHT, roi.WIDTH), dtype=bool)
        refined = roi.refine_pose(corrupted, confidence, frame_confidence, masks)
        keypoints, _, source, origin, _, flags, _, report = refined

        np.testing.assert_allclose(keypoints[view, roi.STRUCTURAL_JOINTS], clean[view, roi.STRUCTURAL_JOINTS])
        self.assertIn(view, report["rotation_views"])
        self.assertTrue(flags[view] & roi.ROTATION_CORRECTION)
        self.assertEqual(source[view, roi.LEFT_HIP], roi.KEYPOINT_SWAPPED)
        self.assertEqual(origin[view, roi.LEFT_HIP], roi.RIGHT_HIP)

    def test_interpolation_respects_mask_and_wraparound(self):
        keypoints = np.zeros((roi.VIEW, roi.KEYPOINT_COUNT, 2), dtype=np.float32)
        confidence = np.zeros((roi.VIEW, roi.KEYPOINT_COUNT), dtype=np.float32)
        source = np.zeros((roi.VIEW, roi.KEYPOINT_COUNT), dtype=np.uint8)
        origin = np.full((roi.VIEW, roi.KEYPOINT_COUNT), -1, dtype=np.int8)
        joint = roi.LEFT_WRIST
        keypoints[:, joint] = [100, 100]
        confidence[:, joint] = 0.9
        source[:, joint] = roi.KEYPOINT_OBSERVED
        origin[:, joint] = joint
        confidence[0, joint] = 0
        source[0, joint] = roi.KEYPOINT_INVALID
        origin[0, joint] = -1
        keypoints[15, joint] = [100, 100]
        keypoints[1, joint] = [120, 100]

        masks = np.zeros((roi.VIEW, roi.HEIGHT, roi.WIDTH), dtype=bool)
        result = roi.interpolate_isolated_keypoints(keypoints, confidence, source, origin, masks)
        self.assertEqual(result[-1], [])

        masks[0, 95:106, 105:116] = True
        result = roi.interpolate_isolated_keypoints(keypoints, confidence, source, origin, masks)
        refined, refined_confidence, refined_source, refined_origin, views = result
        self.assertEqual(views, [0])
        np.testing.assert_allclose(refined[0, joint], [110, 100])
        self.assertEqual(refined_confidence[0, joint], 0.4)
        self.assertEqual(refined_source[0, joint], roi.KEYPOINT_INTERPOLATED)
        self.assertEqual(refined_origin[0, joint], -1)


class ArtifactLifecycleTest(unittest.TestCase):
    def test_finalize_sorts_scan_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            pd.DataFrame({"scan_id": ["scan_b", "scan_a"], "aps_path": ["b.aps", "a.aps"]}).to_csv(dataset, index=False)
            checkpoints = root / "output" / "checkpoints"
            checkpoint_a = empty_checkpoint("scan_a")
            checkpoint_b = empty_checkpoint("scan_b")
            checkpoint_a["frame_confidence"][0] = 0.1
            checkpoint_b["frame_confidence"][0] = 0.2
            checkpoint_a["pose_report"] = np.array(json.dumps({**empty_report(), "detected_views": 1}))
            checkpoint_b["pose_report"] = np.array(json.dumps({**empty_report(), "detected_views": 1}))
            roi.save_checkpoint(checkpoints / "scan_a.npz", checkpoint_a)
            roi.save_checkpoint(checkpoints / "scan_b.npz", checkpoint_b)

            outfile = roi.finalize_artifacts(dataset, root / "output")

            with np.load(outfile, allow_pickle=False) as artifact:
                self.assertEqual(artifact["scan_ids"].tolist(), ["scan_a", "scan_b"])
                np.testing.assert_allclose(artifact["frame_confidence"][:, 0], [0.1, 0.2])
                self.assertEqual(roi.validate_aggregate(artifact, ["scan_a", "scan_b"]), [])

    def test_finalize_rejects_mixed_model_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            pd.DataFrame({"scan_id": ["scan_a", "scan_b"], "aps_path": ["a.aps", "b.aps"]}).to_csv(dataset, index=False)
            checkpoints = root / "output" / "checkpoints"
            roi.save_checkpoint(checkpoints / "scan_a.npz", empty_checkpoint("scan_a"))
            roi.save_checkpoint(
                checkpoints / "scan_b.npz",
                empty_checkpoint("scan_b", model_hash="c" * 64),
            )

            with self.assertRaisesRegex(ValueError, "different pose model"):
                roi.finalize_artifacts(dataset, root / "output")

    def test_shards_are_disjoint_and_cover_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            ids = list("gfedcba")
            pd.DataFrame({"scan_id": ids, "aps_path": [f"{scan_id}.aps" for scan_id in ids]}).to_csv(
                dataset, index=False
            )
            model = root / "pose.pt"
            model.write_bytes(b"pose")
            processed = []
            by_shard = []

            def process(_, scan_id, aps_path, signature, model_hash):
                processed.append((scan_id, str(aps_path)))
                return empty_checkpoint(scan_id, signature, model_hash)

            with (
                patch.object(roi, "YOLO", return_value=object()),
                patch.object(roi, "process_scan", side_effect=process),
                patch.object(roi, "save_checkpoint"),
            ):
                for shard in range(3):
                    start = len(processed)
                    roi.export_artifacts(
                        dataset,
                        root,
                        model,
                        root / f"output_{shard}",
                        shard,
                        3,
                    )
                    by_shard.append(set(scan_id for scan_id, _ in processed[start:]))

            self.assertEqual(set.union(*by_shard), set(ids))
            self.assertEqual(by_shard, [{"a", "d", "g"}, {"b", "e"}, {"c", "f"}])
            self.assertFalse(by_shard[0] & by_shard[1])
            self.assertFalse(by_shard[0] & by_shard[2])
            self.assertFalse(by_shard[1] & by_shard[2])


class GeometryTest(unittest.TestCase):
    def test_half_open_box_and_polygon_clipping(self):
        np.testing.assert_array_equal(roi.clip_box((-4.2, -3.1, 513.2, 661.1)), [0, 0, 512, 660])
        polygon = np.array([[-2, 10], [8, 10], [8, 20], [-2, 20]], dtype=np.float32)
        returned, ratio = roi.clip_polygon(polygon)
        np.testing.assert_array_equal(returned, polygon)
        self.assertAlmostEqual(ratio, 0.8, places=2)

    def test_no_pose_relative_fallback_contract(self):
        images = np.zeros((roi.VIEW, roi.HEIGHT, roi.WIDTH, 3), dtype=np.uint8)
        masks = np.ones((roi.VIEW, roi.HEIGHT, roi.WIDTH), dtype=bool)
        keypoints = np.zeros((roi.VIEW, roi.KEYPOINT_COUNT, 2), dtype=np.float32)
        confidence = np.zeros((roi.VIEW, roi.KEYPOINT_COUNT), dtype=np.float32)
        body_boxes = np.tile([0, 0, roi.WIDTH, roi.HEIGHT], (roi.VIEW, 1)).astype(np.int16)
        result = roi.build_rois(images, masks, keypoints, confidence, body_boxes)
        self.assertEqual(int(result["roi_valid"].sum()), 250)
        self.assertTrue((result["roi_source"][result["roi_valid"]] == roi.ROI_RELATIVE).all())
        self.assertTrue(np.array_equal(result["roi_visible"], result["roi_visibility"] >= 0.5))


class EvaluationContractTest(unittest.TestCase):
    def test_source_hash_is_part_of_pipeline_identity(self):
        checkpoint = empty_checkpoint()
        checkpoint["source_sha256"] = np.array("c" * 64)
        errors = roi.validate_checkpoint(checkpoint)
        self.assertTrue(any("model and source hashes" in error for error in errors))

    def test_annotation_split_is_frozen_in_provenance(self):
        rows = []

        for scan_id, evaluation_set in [("scan_a", "calibration"), ("scan_b", "locked")]:
            for view in range(roi.VIEW):
                rows.append(
                    {
                        "scan_id": scan_id,
                        "view_index": view,
                        "image_file": f"images/{evaluation_set}/{scan_id}_v{view:02d}.png",
                        "difficulty_quartile": 0,
                        "roi_difficulty_quartile": 0,
                        "evaluation_set": evaluation_set,
                        "annotate_pose": True,
                        "annotate_roi": view % 2 == 0,
                    }
                )

        manifest = pd.DataFrame(rows)
        provenance = {"annotation_contract_sha256": roi.annotation_contract_hash(manifest)}
        swapped = manifest.copy()
        swapped["evaluation_set"] = swapped["evaluation_set"].map({"calibration": "locked", "locked": "calibration"})
        swapped["image_file"] = swapped.apply(
            lambda row: f"images/{row.evaluation_set}/{row.scan_id}_v{row.view_index:02d}.png",
            axis=1,
        )

        with self.assertRaisesRegex(ValueError, "frozen GT provenance"):
            roi.validate_ground_truth_provenance(swapped, provenance)

    def test_pilot_sample_cannot_issue_final_gate(self):
        pose = pd.DataFrame(
            {
                "scan_id": np.repeat(["scan_a", "scan_b"], 10),
                "evaluation_set": "locked",
                "visibility": "V",
            }
        )
        rois = pose.copy()
        status = roi.evaluation_sample_status(pose, rois)
        self.assertTrue(status["pilot_only"])
        self.assertFalse(status["pose_ready"])
        self.assertFalse(status["roi_ready"])

    def test_quality_calibration_uses_locked_only(self):
        data = pd.DataFrame(
            [
                {"candidate": "selected", "evaluation_set": "locked", "quality": 0.0, "iou": 1.0},
                {"candidate": "selected", "evaluation_set": "locked", "quality": 1.0, "iou": 0.0},
                {"candidate": "selected", "evaluation_set": "calibration", "quality": 0.0, "iou": 0.0},
                {"candidate": "selected", "evaluation_set": "calibration", "quality": 1.0, "iou": 1.0},
            ]
        )
        quality = roi.roi_quality_calibration(data)
        self.assertEqual(quality["count"], 2)
        self.assertAlmostEqual(quality["spearman"], -1.0)
        self.assertAlmostEqual(quality["auroc_iou_050"], 0.0)

    def test_roi_decision_ignores_calibration_false_activation(self):
        base = {
            "candidate": "selected",
            "evaluation_set": "locked",
            "visibility_scope": "V+I",
            "source": "all",
            "median_iou": 0.8,
            "median_coverage": 0.9,
            "catastrophic_020": 0.0,
        }
        summary = pd.DataFrame(
            [{**base, "zone_group": group} for group in ["all", *sorted(set(roi.ROI_ZONE_GROUPS.values()))]]
        )
        errors = pd.DataFrame(
            [
                {
                    "candidate": "selected",
                    "evaluation_set": "locked",
                    "zone_id": 5,
                    "visibility": "N",
                    "predicted_visible": False,
                    "source": "torso",
                    "scan_id": "locked",
                    "view_index": 0,
                    "iou": np.nan,
                    "coverage": np.nan,
                },
                {
                    "candidate": "selected",
                    "evaluation_set": "calibration",
                    "zone_id": 5,
                    "visibility": "N",
                    "predicted_visible": True,
                    "source": "torso",
                    "scan_id": "calibration",
                    "view_index": 0,
                    "iou": np.nan,
                    "coverage": np.nan,
                },
            ]
        )
        decision = roi.roi_decisions(errors, summary, {"spearman": 0.8, "auroc_iou_050": 0.8}, True)
        self.assertTrue(decision["phase26_geometry_gate"])
        self.assertEqual(decision["opposite_torso_false_activations"], 0)

    def test_pose_decision_rejects_nan_joint_family(self):
        rows = []

        for model, median, pck, catastrophic in [("raw", 10.0, 0.80, 0.20), ("refined", 9.0, 0.85, 0.10)]:
            for group in ["all", *sorted(set(roi.POSE_JOINT_GROUPS.values()))]:
                missing_family = model == "refined" and group == "wrist_ankle"
                rows.append(
                    {
                        "model": model,
                        "evaluation_set": "locked",
                        "visibility_scope": "V",
                        "joint_group": group,
                        "median_error": np.nan if missing_family else median,
                        "median_normalized_error": 0.05,
                        "p90_error": 20.0,
                        "pck_20": pck,
                        "catastrophic_30": catastrophic,
                        "catastrophic_50": 0.05,
                        "coverage": 0.0 if missing_family else 1.0,
                    }
                )

        decisions = roi.pose_decisions(
            pd.DataFrame(rows),
            [
                {"model": "raw", "swap_rate": 0.01},
                {"model": "refined", "swap_rate": 0.01},
            ],
            [{"second": "refined", "ci_high": 0.0}],
            True,
        )
        refined = next(row for row in decisions if row["model"] == "refined")
        self.assertFalse(refined["absolute_pose_gate"])
        self.assertFalse(refined["relative_to_raw_gate"])
        self.assertFalse(refined["joint_families_preserved"])


if __name__ == "__main__":
    unittest.main()
