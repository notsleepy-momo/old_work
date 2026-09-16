import json
import inspect
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from evaluation import EvaluationResult, FurnitureMatchResult, evaluate_single
import pipeline as pipeline_module
from pipeline import Pipeline, normalize_trajectory_durations
from baselines import ablation_experiments, compare_baselines, vlm_baseline
from baselines.compare_baselines import METHODS_ORDER, _build_comparison_lines
from baselines.plot_comparison import parse_comparison_file
from baselines.vlm_baseline import (
    load_devices_for_prompt,
    run_cot,
    run_zeroshot,
)
from llm_config import DEFAULT_LLM_MODEL, require_shared_model, resolve_api_key
from agents.data_models import (
    FurnitureNaming,
    FurnitureNamingGenerationResult,
    RoomAnalysis,
)
from agents.furniture_detection_agent import FurnitureDetectionAgent
from agents.furniture_naming_agent import FurnitureNamingAgent
from agents.prompts import build_furniture_naming_prompt
from agents.room_agent import RoomAgent


def make_match(*, iou, name_correct, area):
    return FurnitureMatchResult(
        gt_name="bed",
        pred_name="bed" if name_correct else "sofa",
        gt_room="bedroom",
        pred_room="bedroom",
        iou=iou,
        name_correct=name_correct,
        gt_norm="bed",
        pred_norm="bed" if name_correct else "sofa",
        gt_area=area,
        centroid_dist=0.0,
    )


class EvaluationProtocolTests(unittest.TestCase):
    def test_evaluator_keeps_global_and_aligned_protocol_names(self):
        layout = {
            "house": {
                "rooms": [{
                    "name": "bedroom",
                    "position": {"x": 0, "y": 0, "width": 10, "height": 10},
                    "furniture": [],
                }]
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            pred_path = os.path.join(tmp, "pred.yaml")
            gt_path = os.path.join(tmp, "gt.yaml")
            for path in (pred_path, gt_path):
                with open(path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(layout, f)

            result_a, result_b = evaluate_single(pred_path, gt_path, verbose=False)

        self.assertEqual(result_a.strategy_name, "A. 全局匹配 (Global)")
        self.assertEqual(result_b.strategy_name, "B. 房间内对齐匹配 (Aligned)")

    def test_aligned_protocol_translates_each_matched_room_pair(self):
        def make_layout(room_x, furniture_x):
            return {
                "house": {
                    "rooms": [{
                        "name": "bedroom",
                        "position": {
                            "x": room_x, "y": 0,
                            "width": 10, "height": 10,
                        },
                        "furniture": [{
                            "name": "bed",
                            "position": {
                                "x": furniture_x, "y": 1,
                                "width": 1, "height": 1,
                            },
                        }],
                    }]
                }
            }

        with tempfile.TemporaryDirectory() as tmp:
            pred_path = os.path.join(tmp, "pred.yaml")
            gt_path = os.path.join(tmp, "gt.yaml")
            with open(pred_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(make_layout(room_x=2, furniture_x=3), f)
            with open(gt_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(make_layout(room_x=0, furniture_x=1), f)

            result_a, result_b = evaluate_single(pred_path, gt_path, verbose=False)

        self.assertEqual(result_a.furniture_f1, 0.0)
        self.assertEqual(result_b.furniture_f1, 1.0)
        self.assertEqual(result_b.furniture_matches[0].offset_applied, (-2.0, 0.0))

    def test_room_metrics_penalize_missing_gt_rooms(self):
        result = EvaluationResult("test")
        result.n_gt_rooms = 2
        result.n_pred_rooms = 1
        result.n_matched_rooms = 1
        result.n_correct_room_type = 1

        result.compute_metrics()

        self.assertAlmostEqual(result.room_type_accuracy, 0.5)
        self.assertAlmostEqual(result.room_precision, 1.0)
        self.assertAlmostEqual(result.room_recall, 0.5)
        self.assertAlmostEqual(result.room_f1, 2 / 3)

    def test_semantic_f1_is_end_to_end(self):
        result = EvaluationResult("test")
        result.n_gt_furniture = 2
        result.n_pred_furniture = 3
        result.gt_furniture_areas = [100, 100]
        result.furniture_matches = [
            make_match(iou=0.8, name_correct=True, area=100),
            make_match(iou=0.7, name_correct=False, area=100),
        ]

        result.compute_metrics()

        self.assertEqual(result.tp_furniture, 2)
        self.assertEqual(result.tp_correct_name, 1)
        self.assertAlmostEqual(result.furniture_naming_accuracy, 0.5)
        self.assertAlmostEqual(result.semantic_precision, 1 / 3)
        self.assertAlmostEqual(result.semantic_recall, 1 / 2)
        self.assertAlmostEqual(result.semantic_f1, 0.4)

    def test_area_strata_include_unmatched_gt_furniture(self):
        result = EvaluationResult("test")
        result.n_gt_furniture = 3
        result.n_pred_furniture = 1
        result.gt_furniture_areas = [25, 100, 300]
        result.furniture_matches = [make_match(iou=0.9, name_correct=True, area=25)]

        result.compute_metrics()

        self.assertEqual(result.per_class['small (<50)']['gt'], 1)
        self.assertEqual(result.per_class['medium (50-200)']['gt'], 1)
        self.assertEqual(result.per_class['large (>=200)']['gt'], 1)
        self.assertEqual(result.per_class['medium (50-200)']['tp'], 0)
        self.assertEqual(result.per_class['large (>=200)']['tp'], 0)


class TrajectoryProtocolTests(unittest.TestCase):
    def test_duration_normalization_uses_seconds_without_mutating_input(self):
        source = [
            {
                "duration": 0.5,
                "start_time": "2025-05-31 05:30:00",
                "end_time": "2025-05-31 05:30:30",
            },
            {
                "duration": 30,
                "duration_unit": "seconds",
                "start_time": "2025-05-31 05:31:00",
                "end_time": "2025-05-31 05:31:30",
            },
            {"duration": 2},
        ]

        normalized = normalize_trajectory_durations(source)

        self.assertEqual([p["duration"] for p in normalized], [30.0, 30.0, 120.0])
        self.assertTrue(all(p["duration_unit"] == "seconds" for p in normalized))
        self.assertEqual(source[0]["duration"], 0.5)

    def test_skip_trajectory_removes_input_from_detection_and_downstream(self):
        class FakeFurnitureAgent:
            trajectory_path = "not-called"

            def process(self, image_path, seg_yaml_path, output_path, **kwargs):
                self.trajectory_path = kwargs["trajectory_json_path"]
                with open(output_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump({"house": {"rooms": []}}, f)

        class FakeConverter:
            @staticmethod
            def convert_layout_to_world(layout):
                return layout

        with tempfile.TemporaryDirectory() as tmp:
            seg_path = os.path.join(tmp, "seg.yaml")
            trajectory_path = os.path.join(tmp, "trajectory.json")
            with open(seg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"house": {"rooms": []}}, f)
            with open(trajectory_path, "w", encoding="utf-8") as f:
                json.dump([{"duration": 1}], f)

            pipeline = Pipeline.__new__(Pipeline)
            pipeline.ablation = {"skip_trajectory": True}
            pipeline.code = "test"
            pipeline.furniture_agent = FakeFurnitureAgent()

            with patch("pipeline.CoordinateConverter.from_yaml", return_value=FakeConverter()):
                _, trajectory_data = pipeline.step2_furniture_and_devices(
                    image_path="",
                    seg_yaml_path=seg_path,
                    trajectory_path=trajectory_path,
                    smart_device_path=None,
                    hatch_mask_path=None,
                    output_dir=tmp,
                )

            self.assertIsNone(pipeline.furniture_agent.trajectory_path)
            self.assertEqual(trajectory_data, [])


class AblationIsolationTests(unittest.TestCase):
    def test_primary_ablation_groups_change_only_declared_boundaries(self):
        self.assertEqual(
            ablation_experiments.MODAL_VARIANTS,
            {
                "Layout Only": {
                    "skip_trajectory": True,
                    "skip_smart_device": True,
                },
                "+ Device": {"skip_trajectory": True},
                "+ Trajectory": {"skip_smart_device": True},
                "Full (All Modalities)": {},
            },
        )
        self.assertEqual(
            ablation_experiments.HYBRID_VARIANTS,
            {
                "CV-only": {"skip_llm_correction": True},
                "Free-form LLM": {"use_free_form_llm": True},
                "Direct LLM Generation": {"use_direct_llm_gen": True},
                "Constrained (Ours)": {},
            },
        )
        self.assertEqual(
            ablation_experiments.DETECTION_VARIANTS,
            {
                "Full (Dual-source)": {},
                "No Hatch": {"skip_hatch": True},
                "No Trajectory Loop": {"skip_trajectory_loop": True},
            },
        )
        self.assertEqual(
            ablation_experiments.REASONING_VARIANTS,
            {
                "Full (Reasoning)": {},
                "No 7-Layer Priority": {"skip_layered_priority": True},
                "No Behavior Prior": {"skip_behavior_prior": True},
                "No Allowed List": {"skip_allowed_list": True},
                "No Shape Constraint": {"skip_shape_constraint": True},
            },
        )

    def test_variant_order_puts_each_group_baseline_first(self):
        self.assertEqual(
            ablation_experiments._ordered_variant_names(
                ablation_experiments.MODAL_VARIANTS),
            ["Full (All Modalities)", "Layout Only", "+ Device", "+ Trajectory"],
        )
        self.assertEqual(
            ablation_experiments._ordered_variant_names(
                ablation_experiments.HYBRID_VARIANTS)[0],
            "Constrained (Ours)",
        )
        self.assertEqual(
            ablation_experiments._ordered_variant_names(
                ablation_experiments.DETECTION_VARIANTS)[0],
            "Full (Dual-source)",
        )

    def test_group_selection_deduplicates_and_keeps_module_supplement_explicit(self):
        selected = ablation_experiments._resolve_group_infos(
            ["modal", "modal"])
        self.assertEqual([key for key, _ in selected], ["modal"])

        selected = ablation_experiments._resolve_group_infos(
            ["all", "module"])
        self.assertEqual(
            [key for key, _ in selected],
            ["module", "modal", "hybrid", "detection", "reasoning"],
        )

    def test_direct_generation_parser_does_not_require_cv_candidates(self):
        agent = FurnitureDetectionAgent.__new__(FurnitureDetectionAgent)
        result = agent._parse_direct_furniture_generation(
            json.dumps([
                {"action": "keep", "bbox": {
                    "x": 10, "y": 20, "width": 30, "height": 40,
                }},
            ]),
            "room1", crop_left=100, crop_top=200,
            crop_w=80, crop_h=90,
        )
        self.assertEqual(result, [("room1", 110.0, 220.0, 30.0, 40.0)])

    def test_direct_generation_does_not_fallback_to_cv_candidates(self):
        agent = FurnitureDetectionAgent.__new__(FurnitureDetectionAgent)
        agent._ablation = {"use_direct_llm_gen": True}
        self.assertEqual(
            agent._parse_direct_furniture_generation(
                "not json", "room1", 0, 0, 100, 100),
            [],
        )

    def test_free_form_parser_accepts_add_and_delete_lists(self):
        agent = FurnitureDetectionAgent.__new__(FurnitureDetectionAgent)
        agent._ablation = {"use_free_form_llm": True}
        candidates = [
            {
                "candidate_id": 1,
                "candidate_type": "hatch_furniture",
                "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
            },
            {
                "candidate_id": 2,
                "candidate_type": "trajectory_surrounded",
                "bbox": {"x": 40, "y": 10, "width": 20, "height": 20},
            },
        ]
        result = agent._parse_room_furniture_actions(
            json.dumps([
                {"action": "delete", "candidate_ids": [1, 2]},
                {"action": "add", "bbox": {
                    "x": 70, "y": 15, "width": 12, "height": 10,
                }},
            ]),
            candidates, "room1", 0, 0, 0, 0, 100, 100,
            allow_unrestricted=True,
        )
        self.assertEqual(result, [("room1", 70.0, 15.0, 12.0, 10.0)])

    def test_constrained_parser_protects_red_candidate_from_delete(self):
        agent = FurnitureDetectionAgent.__new__(FurnitureDetectionAgent)
        candidates = [{
            "candidate_id": 1,
            "candidate_type": "hatch_furniture",
            "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
        }]
        result = agent._parse_room_furniture_actions(
            json.dumps([{"action": "delete", "candidate_id": 1}]),
            candidates, "room1", 0, 0, 0, 0, 100, 100,
        )
        self.assertEqual(result, [("room1", 10, 10, 20, 20)])

    def test_reasoning_prompt_variants_remove_the_declared_constraints(self):
        no_allowed = build_furniture_naming_prompt(skip_allowed_list=True)
        self.assertIn("ALLOWED FURNITURE LISTS (ABLATION: DISABLED)", no_allowed)
        self.assertNotIn("Pick furniture names **STRICTLY from the allowed list", no_allowed)
        self.assertNotIn("Allowed List Only", no_allowed)

        no_shape = build_furniture_naming_prompt(skip_shape_constraint=True)
        self.assertIn("Size + Shape Filter (ABLATION: DISABLED)", no_shape)
        self.assertNotIn("For each furniture with position `(x, y, width, height)`, calculate:", no_shape)
        self.assertNotIn("Sofa's LONG SIDE MUST touch a wall", no_shape)

    def test_no_behavior_prior_is_preserved_across_room_retries(self):
        agent = RoomAgent.__new__(RoomAgent)
        room_analysis = RoomAnalysis(
            room_id="room1", room_type="bedroom", confidence=0.8)
        agent.generate_room_analyses = Mock(
            return_value=SimpleNamespace(room_analyses=[room_analysis]))
        agent._apply_partial_priority = Mock(side_effect=lambda _, analyses: analyses)
        agent._apply_layered_priority = Mock(
            side_effect=AssertionError("full priority reintroduced"))
        agent.validate_room_analyses = Mock(side_effect=[
            SimpleNamespace(is_valid=False, errors=["retry"]),
            SimpleNamespace(is_valid=False, errors=["retry"]),
            SimpleNamespace(is_valid=False, errors=["retry"]),
            SimpleNamespace(is_valid=False, errors=["retry"]),
            SimpleNamespace(is_valid=True, errors=[]),
        ])
        agent.fix_room_analyses = Mock(side_effect=lambda analyses, _: analyses)

        result = agent.analyze_rooms(
            {"house": {"rooms": [{"id": "room1", "furniture": []}]}},
            trajectory_data=[],
            ablation={"skip_behavior_prior": True},
        )

        self.assertEqual(result[0].room_id, "room1")
        self.assertEqual(agent._apply_partial_priority.call_count, 2)
        agent._apply_layered_priority.assert_not_called()

    def test_formal_variants_each_disable_the_intended_boundary(self):
        self.assertEqual(
            ablation_experiments.PIPELINE_VARIANTS,
            {
                "Full (Ours)": {},
                "No Trajectory-Loop Recall": {
                    "skip_trajectory_loop": True,
                },
                "No LLM Geometry Correction": {
                    "skip_llm_correction": True,
                },
                "No 7-Layer Room Priority": {
                    "skip_layered_priority": True,
                },
                "No Behavior Agent": {
                    "skip_behavior_module": True,
                },
                "No Naming Funnel": {
                    "skip_naming_funnel": True,
                },
            },
        )

    def test_no_llm_variant_keeps_deterministic_candidate_cleanup(self):
        agent = FurnitureDetectionAgent.__new__(FurnitureDetectionAgent)
        agent._annotate_structure_features = Mock()
        agent._auto_merge_strong_signal_pairs = Mock(side_effect=lambda items, _: items)
        candidates = [
            {
                "candidate_id": 0,
                "candidate_type": "trajectory_surrounded",
                "bbox": {"x": 0, "y": 0, "width": 2, "height": 2},
            },
            {
                "candidate_id": 1,
                "candidate_type": "hatch_furniture",
                "bbox": {"x": 5, "y": 5, "width": 20, "height": 20},
            },
        ]

        result = agent._prepare_candidates_for_correction(
            candidates, traj_mask=None, scale=1.0, room_name="room1")

        agent._annotate_structure_features.assert_called_once_with(
            candidates, None, 1.0)
        self.assertEqual([candidate["candidate_id"] for candidate in result], [1])
        agent._auto_merge_strong_signal_pairs.assert_called_once_with(
            result, "room1")

    def test_ablation_summary_reports_both_matching_protocols(self):
        metrics = SimpleNamespace(
            room_f1=0.8,
            furniture_f1=0.7,
            semantic_f1=0.6,
            furniture_naming_accuracy=0.5,
            semantic_precision=0.75,
        )
        lines = "\n".join(ablation_experiments._build_module_table(
            "module",
            {"Full (Ours)": {}},
            {"Full (Ours)": {"result_a": metrics, "result_b": metrics}},
        ))

        self.assertIn("Protocol: A. Global matching", lines)
        self.assertIn("Protocol: B. Room-aligned matching", lines)

    def test_naming_funnel_variant_uses_unconstrained_prompt(self):
        agent = FurnitureNamingAgent.__new__(FurnitureNamingAgent)
        agent._output_dir = None
        agent.chain = Mock()
        agent.unconstrained_chain = Mock()
        agent.unconstrained_chain.invoke.return_value = SimpleNamespace(
            content=json.dumps([{
                "furniture_id": "room1_fur1",
                "name": "lamp",
                "room_id": "room1",
                "confidence": 0.7,
            }])
        )
        layout = {
            "house": {
                "rooms": [{
                    "id": "room1",
                    "position": {"x": 0, "y": 0, "width": 10, "height": 10},
                    "furniture": [{
                        "id": "room1_fur1",
                        "position": {"x": 2, "y": 2, "width": 3, "height": 4},
                    }],
                }],
            },
        }
        room_analyses = [RoomAnalysis(
            room_id="room1", room_type="bedroom", confidence=0.8)]

        result = agent.generate_furniture_namings(
            layout, room_analyses, [], ablation={"skip_naming_funnel": True})

        self.assertEqual(result.furniture_namings[0].name, "lamp")
        agent.unconstrained_chain.invoke.assert_called_once()
        agent.chain.invoke.assert_not_called()
        payload = agent.unconstrained_chain.invoke.call_args.args[0]
        self.assertNotIn("structured_features", payload)
        self.assertNotIn("_features", payload["house_layout_yaml"])

    def test_naming_funnel_variant_skips_allowed_list_and_geometry_repairs(self):
        agent = FurnitureNamingAgent.__new__(FurnitureNamingAgent)
        agent._ablation = {"skip_naming_funnel": True}
        agent._apply_hard_geometry_rules = Mock(side_effect=AssertionError)
        layout = {
            "house": {
                "rooms": [{
                    "id": "room1",
                    "position": {"x": 0, "y": 0, "width": 10, "height": 10},
                    "furniture": [{
                        "id": "room1_fur1",
                        "position": {"x": 0, "y": 0, "width": 8, "height": 2},
                    }],
                }],
            },
        }
        naming = FurnitureNaming(
            furniture_id="room1_fur1", name="lamp", room_id="room1", confidence=0.8)
        agent.generate_furniture_namings = Mock(
            return_value=FurnitureNamingGenerationResult(
                furniture_namings=[naming], confidence=0.8))
        room_analyses = [RoomAnalysis(
            room_id="room1", room_type="living_room", confidence=0.8)]

        result = agent.name_furniture(layout, room_analyses, [])

        self.assertEqual(result[0].name, "lamp")
        agent._apply_hard_geometry_rules.assert_not_called()


class VlmInputProtocolTests(unittest.TestCase):
    def test_device_prompt_uses_per_home_input_coordinates(self):
        device_input = {
            "house": {
                "size": {"x": 63, "y": 137},
                "furniture": [
                    {"name": "Oven", "position": {"x": 57, "y": 126}},
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "devices.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(device_input, f)

            devices_text, house_info = load_devices_for_prompt(path)

        self.assertIn("Oven: at (57, 126)", devices_text)
        self.assertIn("63 dm wide", house_info)
        self.assertIn("137 dm tall", house_info)

    def test_vlm_generation_interfaces_do_not_accept_gt_paths(self):
        self.assertNotIn("gt_path", inspect.signature(run_zeroshot).parameters)
        self.assertNotIn("gt_path", inspect.signature(run_cot).parameters)
        self.assertNotIn("'--gt'", inspect.getsource(compare_baselines.main))
        self.assertNotIn("'--gt'", inspect.getsource(vlm_baseline.main))

    def test_default_model_and_jpeg_data_url_are_consistent(self):
        from PIL import Image

        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok"))])
        create = Mock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "map.png")
            Image.new("RGB", (8, 8), "white").save(image_path)
            with patch("openai.OpenAI", return_value=client):
                vlm_baseline.call_vlm(
                    "system", "user", image_path, api_key="test-key")

        self.assertEqual(vlm_baseline.DEFAULT_MODEL, "gpt-5.6-sol")
        request = create.call_args.kwargs
        image_url = request["messages"][1]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))

    def test_cot_naming_step_cannot_replace_room_or_furniture_geometry(self):
        from PIL import Image

        step1 = {
            "house_size": {"x": 20, "y": 30},
            "rooms": [{
                "id": "room_1",
                "name": "bedroom",
                "position": {"x": 2, "y": 3, "width": 10, "height": 12},
                "doors": [{"id": "door_1", "x": 5, "y": 3}],
            }],
        }
        step2 = {
            "rooms": [{
                "room_id": "room_1",
                "furniture": [{
                    "id": "room_1_fur1",
                    "name": "unknown",
                    "position": {"x": 4, "y": 5, "width": 6, "height": 7},
                }],
            }],
        }
        step3 = {
            "furniture_names": [{"id": "room_1_fur1", "name": "bed"}],
            "rooms": [{
                "id": "room_1",
                "position": {"x": 0, "y": 0, "width": 1, "height": 1},
                "furniture": [{
                    "id": "room_1_fur1",
                    "position": {"x": 0, "y": 0, "width": 1, "height": 1},
                }],
            }],
        }

        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "room_0001.png")
            devices_path = os.path.join(tmp, "devices.yaml")
            output_path = os.path.join(tmp, "prediction.yaml")
            Image.new("RGB", (16, 16), "white").save(image_path)
            with open(devices_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"house": {"size": {"x": 20, "y": 30},
                                          "furniture": []}}, f)

            with patch.object(vlm_baseline, "call_vlm",
                              side_effect=[json.dumps(step1), json.dumps(step2)]), \
                    patch.object(vlm_baseline, "call_vlm_text_only",
                                 return_value=json.dumps(step3)):
                result = run_cot(
                    image_path, devices_path, None, output_path=output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                saved = yaml.safe_load(f)

        room = result["layout"]["house"]["rooms"][0]
        furniture = room["furniture"][0]
        self.assertEqual(room["position"],
                         {"x": 2.0, "y": 3.0, "width": 10.0, "height": 12.0})
        self.assertEqual(furniture["position"],
                         {"x": 4.0, "y": 5.0, "width": 6.0, "height": 7.0})
        self.assertEqual(furniture["name"], "bed")
        self.assertEqual(saved, result["layout"])

    def test_invalid_zeroshot_response_does_not_overwrite_previous_output(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "room_0001.png")
            devices_path = os.path.join(tmp, "devices.yaml")
            output_path = os.path.join(tmp, "prediction.yaml")
            Image.new("RGB", (8, 8), "white").save(image_path)
            with open(devices_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"house": {"size": {"x": 10, "y": 10},
                                          "furniture": []}}, f)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("previous-valid-result\n")

            with patch.object(vlm_baseline, "call_vlm", return_value="house: []"):
                with self.assertRaises(vlm_baseline.VlmBaselineError):
                    run_zeroshot(
                        image_path, devices_path, None, output_path=output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

        self.assertEqual(content, "previous-valid-result\n")

    def test_observed_device_xy_is_preserved_but_size_remains_estimated(self):
        layout = {
            "house": {
                "size": {"x": 20, "y": 20},
                "rooms": [{
                    "name": "kitchen",
                    "position": {"x": 0, "y": 0, "width": 20, "height": 20},
                    "furniture": [{
                        "id": "Oven_001",
                        "name": "oven",
                        "position": {"x": 2, "y": 3, "width": 4, "height": 5},
                    }],
                }],
            },
        }
        known = [{"id": "Oven_001", "name": "Oven", "x": 7.0, "y": 8.0}]

        clean = vlm_baseline.validate_layout(
            layout, {"x": 20, "y": 20}, known)
        oven = clean["house"]["rooms"][0]["furniture"][0]

        self.assertEqual(oven["name"], "Oven")
        self.assertEqual(oven["position"],
                         {"x": 7.0, "y": 8.0, "width": 4.0, "height": 5.0})


class VlmGenerationTransactionTests(unittest.TestCase):
    def test_failed_subprocess_never_promotes_or_scores_stale_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = os.path.join(tmp, "candidate.yaml")
            final = os.path.join(tmp, "final.yaml")
            with open(candidate, "w", encoding="utf-8") as f:
                f.write("new-but-failed\n")
            with open(final, "w", encoding="utf-8") as f:
                f.write("old-result\n")

            with patch.object(compare_baselines.subprocess, "run",
                              return_value=SimpleNamespace(returncode=1)):
                success = compare_baselines.generate_vlm_prediction(
                    ["fake-command"], candidate, final)

            with patch.object(compare_baselines, "run_evaluation") as evaluate:
                scored = compare_baselines.evaluate_vlm_prediction(
                    success, final, "gt.yaml", "VLM Zero-shot")

            with open(final, "r", encoding="utf-8") as f:
                content = f.read()

        self.assertFalse(success)
        self.assertEqual(scored, (None, None))
        evaluate.assert_not_called()
        self.assertEqual(content, "old-result\n")


class SharedLlmConfigurationTests(unittest.TestCase):
    def test_pipeline_passes_one_model_and_key_to_every_agent(self):
        agent_names = (
            "FurnitureDetectionAgent",
            "RoomAgent",
            "BehaviorAgent",
            "FurnitureNamingAgent",
        )
        constructors = {name: Mock(name=name) for name in agent_names}

        with patch.dict(os.environ, {
                "FURNITURE_API_KEY": "shared-test-key",
                "FURNITURE_BASE_URL": "https://example.invalid/v1",
        }, clear=True):
            with patch.multiple(pipeline_module, **constructors):
                pipeline_module.Pipeline()

        for constructor in constructors.values():
            constructor.assert_called_once_with(
                model=DEFAULT_LLM_MODEL,
                api_key="shared-test-key",
                base_url="https://example.invalid/v1",
            )

    def test_old_openai_key_is_not_used_as_agent_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "legacy-key"}, clear=True):
            self.assertIsNone(resolve_api_key(required=False))
            with self.assertRaises(ValueError):
                resolve_api_key()

    def test_per_agent_model_drift_is_rejected(self):
        self.assertEqual(require_shared_model(), "gpt-5.6-sol")
        with self.assertRaises(ValueError):
            require_shared_model("another-model")


class ComparisonProtocolTests(unittest.TestCase):
    @staticmethod
    def make_result():
        result = EvaluationResult("test")
        result.room_f1 = 0.8
        result.furniture_f1 = 0.7
        result.semantic_f1 = 0.6
        result.furniture_naming_accuracy = 0.5
        result.tp_correct_name = 4
        return result

    def test_both_protocol_tables_contain_every_method(self):
        results = {
            f"{method}_{protocol}": self.make_result()
            for method, _ in METHODS_ORDER
            for protocol in ("global", "aligned")
        }
        content = "\n".join(_build_comparison_lines(results))

        global_section, aligned_section = content.split(
            "Protocol: B. 房间内对齐匹配 (Aligned)")
        global_section = global_section.split(
            "Protocol: A. 全局匹配 (Global)", maxsplit=1)[1]
        for _, label in METHODS_ORDER:
            self.assertIn(label, global_section)
            self.assertIn(label, aligned_section)
        self.assertNotIn("Oracle", content)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "comparison.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            parsed = parse_comparison_file(path)

        self.assertEqual(set(parsed), {"Global", "Aligned"})
        self.assertEqual(len(parsed["Global"]), len(METHODS_ORDER))
        self.assertEqual(len(parsed["Aligned"]), len(METHODS_ORDER))


class ProtocolViolationTests(unittest.TestCase):
    def test_pre_final_protv_uses_generated_names_before_vocabulary_repair(self):
        generated = [
            FurnitureNaming(
                furniture_id="room1_fur1", name="sofa",
                room_id="room1", confidence=0.8),
            FurnitureNaming(
                furniture_id="room1_fur2", name="bed",
                room_id="room1", confidence=0.8),
            FurnitureNaming(
                furniture_id="room1_fur3", name="oven",
                room_id="room1", confidence=0.95),
        ]
        rooms = [RoomAnalysis(
            room_id="room1", room_type="bedroom", confidence=0.9)]

        with patch.object(ablation_experiments, "FURNITURE_ALLOWED",
                          {"bedroom": ["bed"], "other": []}), \
                patch.object(ablation_experiments, "SMART_DEVICE_NAMES", {"oven"}):
            rate, violations, total = (
                ablation_experiments.compute_pre_final_protocol_violation_rate(
                    generated, rooms))

        self.assertEqual((violations, total), (1, 2))
        self.assertAlmostEqual(rate, 0.5)

    def test_naming_agent_keeps_pre_final_snapshot_before_repair(self):
        agent = FurnitureNamingAgent.__new__(FurnitureNamingAgent)
        agent._ablation = {}
        generated = FurnitureNaming(
            furniture_id="room1_fur1", name="sofa",
            room_id="room1", confidence=0.8)
        repaired = FurnitureNaming(
            furniture_id="room1_fur1", name="bed",
            room_id="room1", confidence=0.5)
        agent.generate_furniture_namings = Mock(
            return_value=FurnitureNamingGenerationResult(
                furniture_namings=[generated], confidence=0.8))
        agent.validate_furniture_namings = Mock(
            return_value=SimpleNamespace(is_valid=False, errors=["vocab"]))
        agent.fix_furniture_namings = Mock(return_value=[repaired])
        room_analyses = [RoomAnalysis(
            room_id="room1", room_type="bedroom", confidence=0.9)]

        final = agent.name_furniture(
            {"house": {"rooms": []}}, room_analyses, [])

        self.assertEqual(agent.last_pre_final_namings[0].name, "sofa")
        self.assertEqual(final[0].name, "bed")

    def test_smart_devices_are_excluded_from_both_denominator_and_numerator(self):
        prediction = {
            "house": {
                "rooms": [{
                    "name": "bedroom",
                    "furniture": [
                        {"name": "TV_Stand"},
                        {"name": "bed"},
                        {"name": "oven"},
                    ],
                }]
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prediction.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(prediction, f)

            with patch.object(ablation_experiments, "FURNITURE_ALLOWED",
                              {"bedroom": ["bed"], "other": []}), \
                    patch.object(ablation_experiments, "SMART_DEVICE_NAMES", {"TV Stand"}):
                rate, violations, total = ablation_experiments.compute_protocol_violation_rate(path)

        self.assertEqual((violations, total), (1, 2))
        self.assertAlmostEqual(rate, 0.5)


if __name__ == "__main__":
    unittest.main()
