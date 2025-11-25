import re
from typing import Dict, Tuple, Optional
import sacrebleu

def compute_bleu(lg_pair, ref, pred):
    # 新增类型检查
    pred = pred if isinstance(pred, str) else ""
    
    src_lang = lg_pair.split("-")[0]
    tgt_lang = lg_pair.split("-")[1]
    
    tokenize = "zh" if tgt_lang == "zh" else "ja-mecab" if tgt_lang == "ja" else "13a"
    
    refs = [[ref]]
    sys = [pred]

    bleu_str = str(sacrebleu.corpus_bleu(sys, refs, tokenize=tokenize))  # 注意bleu tokenize
    bleu_score = re.search(r'BLEU = (\d+\.\d+)', bleu_str).group(1)

    print(f"[BLEU Score] {bleu_score}")
    return float(bleu_score)

def extract_solution(solution_str: str) -> Tuple[str, str, str, str]:
    """Extracts the draft, reflection, and final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (draft_answer, reflection_text, final_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str: # base
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str: # qwen and tower
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str: # llama3
        processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, None, None, solution_str

    # Extract draft translation
    draft_pattern = r'<draft>(.*?)</draft>'
    draft_matches = list(re.finditer(draft_pattern, processed_str, re.DOTALL))
    draft_answer = draft_matches[-1].group(1).strip() if draft_matches else None

    # Extract reflection
    reflection_pattern = r'<reflection>(.*?)</reflection>'
    reflection_matches = list(re.finditer(reflection_pattern, processed_str, re.DOTALL))
    reflection_text = reflection_matches[-1].group(1).strip() if reflection_matches else None

    # Extract final translation
    final_pattern = r'<final>(.*?)</final>'
    final_matches = list(re.finditer(final_pattern, processed_str, re.DOTALL))
    final_answer = final_matches[-1].group(1).strip() if final_matches else None

    if not final_answer:
        print("[Error] No valid answer tags found")
        return None, None, None, processed_str
        
    return draft_answer, reflection_text, final_answer, processed_str


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of Reflex-RL response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags for reflex format
    tags = {
        'draft_start': ('<draft>', 1),
        'draft_end': ('</draft>', 1),
        'reflection_start': ('<reflection>', 1),
        'reflection_end': ('</reflection>', 1),
        'final_start': ('<final>', 1),
        'final_end': ('</final>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order: draft -> reflection -> final
    if validation_passed and all(pos >= 0 for pos in positions.values()):
        if (positions['draft_start'] > positions['draft_end'] or
            positions['draft_end'] > positions['reflection_start'] or
            positions['reflection_start'] > positions['reflection_end'] or
            positions['reflection_end'] > positions['final_start'] or
            positions['final_start'] > positions['final_end']):
            print("  [Error] Incorrect tag order: Expected <draft>...</draft><reflection>...</reflection><final>...</final>")
            validation_passed = False
        else:
            print("  Tag sequence validation passed")

    return validation_passed


def normalize_text_for_comparison(text: str) -> str:
    """标准化文本用于比较draft和final是否相同
    
    Args:
        text: 输入文本
        
    Returns:
        标准化后的文本(去除多余空白、统一小写)
    """
    if not text:
        return ""
    # 去除多余空白,转小写
    return re.sub(r'\s+', ' ', text.strip().lower())


def compute_score(reward_metric: str,
                 reward_type: str,
                 final_metric_score: Optional[float],
                 draft_metric_score: Optional[float],
                 lg_pair: str,
                 bleu_threshold: float,
                 comet_threshold: float,
                 solution_str: str, 
                 ground_truth: str,
                 scale_factor: float = 100.0,
                 check_think: bool = True,
                 format_reward: int = 1,
                 w_final: float = 1.0,
                 w_improve: float = 1.0) -> float:
    """Computes comprehensive score for model response using Reflex-RL framework.
    
    Args:
        solution_str: Raw model response string
        ground_truth: target ground truth data
        final_metric_score: Model-based metric score for final answer
        draft_metric_score: Model-based metric score for draft answer
        w_final: Weight for final quality reward
        w_improve: Weight for improvement gain reward
        
    Returns:
        Total weighted score (r_final + r_improve)
    """
    print("\n" + "="*80)
    print(" Processing Training Sample (Reflex-RL) ".center(80, '='))
    
    # Extract draft, reflection, and final answer for Reflex-RL
    draft_answer, reflection_text, final_answer, processed_str = extract_solution(solution_str)
    print(f"\n[Prompt + Response]\n{solution_str}")

    def is_src_lang(text, src_lang):
        if not text:
            return False

        if src_lang == "zh":
            # 中文源语言
            return bool(re.search(r'[\u4e00-\u9fa5]', text))
        elif src_lang == "ja":
            # 日文源语言
            return bool(re.search(r'[\u3040-\u30ff]', text))
        elif src_lang == "en":
            # 英文源语言，使用简单且稳定的逻辑
            letters = re.findall(r'[a-zA-Z]', text)
            return len(letters) > max(5, len(text) * 0.3)
        else:
            # 其他语言用保守策略
            return False

    src_lang = lg_pair.split("-")[0]

    # Validate response structure  
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)

    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    if draft_answer and is_src_lang(draft_answer, src_lang):
        print(f"[Penalty] Draft is in SOURCE language ({src_lang}). Penalizing heavily.")
        total_score = -3 + format_score
        print("Skipping BLEU/COMET computation due to draft penalty.\n")
        return total_score

    # Initialize reward components
    r_final = 0.0
    r_improve = 0.0

    if format_correct and final_answer:
        # ===== 【新增】检测draft和final是否完全相同 =====
        if draft_answer and final_answer:
            draft_normalized = normalize_text_for_comparison(draft_answer)
            final_normalized = normalize_text_for_comparison(final_answer)
            
            if draft_normalized == final_normalized:
                print("\n" + "!"*80)
                print("[LAZY BEHAVIOR DETECTED] Draft and Final are IDENTICAL!")
                print(f"Draft: {draft_answer}")
                print(f"Final: {final_answer}")
                print("Applying improvement penalty: r_improve = -1")
                print("!"*80)
                r_improve = -1.0
                # 注意:这里仍然需要计算r_final,因为总分 = w_final*r_final + w_improve*r_improve
                # 下面会继续计算r_final,但r_improve已经固定为-1
        
        # ===== 【优化】提前计算所有需要的 BLEU 分数，避免重复计算 =====
        # 根据 reward_metric 决定是否需要计算 BLEU
        need_bleu = reward_metric in ['BLEU', 'Merge']
        
        final_bleu = None
        draft_bleu = None
        
        if need_bleu:
            final_bleu = compute_bleu(lg_pair, ground_truth, final_answer)
            if draft_answer:
                draft_bleu = compute_bleu(lg_pair, ground_truth, draft_answer)
        
        # ===== 1. Final Quality Reward (r_final) =====
        if reward_type == 'discrete':
            if reward_metric == 'BLEU':
                r_final = 2 if final_bleu > bleu_threshold else -1.5
            elif reward_metric == 'Model':
                if final_metric_score is None:
                    raise ValueError("final_metric_score is None, enable comet or cometfree use_rm")
                r_final = 2 if final_metric_score > comet_threshold else -1.5
            elif reward_metric == 'Merge':
                if final_metric_score is None:
                    raise ValueError("final_metric_score is None, enable comet or cometfree use_rm")
                r_final = 2 if (final_bleu > bleu_threshold and final_metric_score > comet_threshold) else -1.5

        elif reward_type == 'continuous':
            if reward_metric == 'BLEU':
                r_final = float(final_bleu) / float(scale_factor)
            elif reward_metric == 'Model':
                if final_metric_score is None:
                    raise ValueError("final_metric_score is None, enable comet or cometfree use_rm")
                r_final = float(final_metric_score) / float(scale_factor)
            elif reward_metric == 'Merge':
                if final_metric_score is None:
                    raise ValueError("final_metric_score is None, enable comet or cometfree use_rm")
                # Merge 模式：取平均或相加（这里用相加后除以 scale_factor）
                r_final = (float(final_bleu) + float(final_metric_score)) / float(scale_factor)

        # ===== 2. Improvement Gain Reward (r_improve) =====
        # 【关键修改】只有在没有检测到偷懒行为时才计算正常的r_improve
        if draft_answer and final_answer and r_improve != -1.0:
            if reward_metric == 'Model':
                # 纯模型评分模式
                if draft_metric_score is not None and final_metric_score is not None:
                    metric_improvement = final_metric_score - draft_metric_score
                    
                    if reward_type == 'continuous':
                        r_improve = metric_improvement / float(scale_factor)
                    elif reward_type == 'discrete':
                        r_improve = 1.0 if metric_improvement > 1.0 else -0.5 if metric_improvement < -1.0 else 0.0
                    
                    print(f"[Improvement Analysis - Model-based]")
                    print(f"Draft Metric Score: {draft_metric_score:.2f}")
                    print(f"Final Metric Score: {final_metric_score:.2f}")
                    print(f"Metric Improvement: {metric_improvement:.2f}")
                else:
                    # 回退到 BLEU（如果没有 draft_metric_score）
                    if draft_bleu is None:
                        draft_bleu = compute_bleu(lg_pair, ground_truth, draft_answer)
                    if final_bleu is None:
                        final_bleu = compute_bleu(lg_pair, ground_truth, final_answer)
                    
                    bleu_improvement = final_bleu - draft_bleu
                    if reward_type == 'continuous':
                        r_improve = bleu_improvement / float(scale_factor)
                    elif reward_type == 'discrete':
                        r_improve = 1.0 if bleu_improvement > 1.0 else -0.5 if bleu_improvement < -1.0 else 0.0
                    
                    print(f"[Improvement Analysis - BLEU-based (fallback)]")
                    print(f"Draft BLEU: {draft_bleu:.2f}")
                    print(f"Final BLEU: {final_bleu:.2f}")
                    print(f"BLEU Improvement: {bleu_improvement:.2f}")
                    
            elif reward_metric == 'Merge':
                # Merge 模式：同时使用 BLEU 和 Model metric
                if draft_metric_score is not None and final_metric_score is not None:
                     # =============================
                    # ⭐ 新增逻辑：draft_metric_score 不达标，禁用 improve reward
                    # =============================
                    if draft_metric_score <= 65: 
                        print("[Improve Disabled] Draft metric <= 70")
                        r_improve += (draft_metric_score - 65) / float(65.0)
                    if draft_bleu <= 20:
                        print("[Improve Disabled] Draft Blue <= 20")
                        r_improve += (draft_bleu - 20) / float(20.0)
                    if draft_metric_score > 65 and draft_bleu > 20:
                        # BLEU 已经提前计算了
                        bleu_improvement = min(1,(final_bleu - draft_bleu) / float(50.0))
                        metric_improvement = min(1,(final_metric_score - draft_metric_score) / float(20.0))
                        if reward_type == 'continuous':
                            if bleu_improvement < 0:
                                bleu_improvement = -1
                            if metric_improvement < 0:
                                metric_improvement = -1
                            # 两种改进相加
                            r_improve = bleu_improvement + metric_improvement
                    
                        elif reward_type == 'discrete':
                            avg_improvement = (bleu_improvement + metric_improvement) / 2.0
                            r_improve = 1.0 if avg_improvement > 1.0 else -0.5 if avg_improvement < -1.0 else 0.0
                        
                        print(f"[Improvement Analysis - Merge Mode]")
                        print(f"Draft BLEU: {draft_bleu:.2f}, Final BLEU: {final_bleu:.2f}, BLEU Improve: {bleu_improvement:.2f}")
                        print(f"Draft Metric: {draft_metric_score:.2f}, Final Metric: {final_metric_score:.2f}, Metric Improve: {metric_improvement:.2f}")
                        print(f"Combined Improvement: {bleu_improvement + metric_improvement:.2f}")
                else:
                    # 回退到纯 BLEU
                    bleu_improvement = final_bleu - draft_bleu
                    if reward_type == 'continuous':
                        r_improve = bleu_improvement / float(scale_factor)
                    elif reward_type == 'discrete':
                        r_improve = 1.0 if bleu_improvement > 1.0 else -0.5 if bleu_improvement < -1.0 else 0.0
                    
                    print(f"[Improvement Analysis - BLEU-based (fallback)]")
                    print(f"Draft BLEU: {draft_bleu:.2f}")
                    print(f"Final BLEU: {final_bleu:.2f}")
                    print(f"BLEU Improvement: {bleu_improvement:.2f}")
                    
            elif reward_metric == 'BLEU':
                # 纯 BLEU 模式（BLEU 已经提前计算）
                bleu_improvement = final_bleu - draft_bleu
                
                if reward_type == 'continuous':
                    r_improve = bleu_improvement / float(scale_factor)
                elif reward_type == 'discrete':
                    r_improve = 1.0 if bleu_improvement > 1.0 else -0.5 if bleu_improvement < -1.0 else 0.0

                print(f"[Improvement Analysis - BLEU-based]")
                print(f"Draft BLEU: {draft_bleu:.2f}")
                print(f"Final BLEU: {final_bleu:.2f}")
                print(f"BLEU Improvement: {bleu_improvement:.2f}")

        # ===== 打印内容验证信息 =====
        print(f"\n[Content Validation]")
        print(f"Reference: {ground_truth}")
        print(f"Draft: {draft_answer}")
        print(f"Final: {final_answer}")
        
        # 打印评分信息
        if final_bleu is not None:
            print(f"Final BLEU: {final_bleu:.2f}")
        if final_metric_score is not None:
            print(f"Final Metric Score: {final_metric_score:.2f}")
        if draft_bleu is not None:
            print(f"Draft BLEU: {draft_bleu:.2f}")
        if draft_metric_score is not None:
            print(f"Draft Metric Score: {draft_metric_score:.2f}")
    else:
        r_final = -2
        r_improve = -1  
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    # Compute total weighted reward
    total_score = w_final * r_final + w_improve * r_improve + format_score
    
    print("\n" + "-"*80)
    print(f" Reflex-RL Reward Breakdown ".center(80, '-'))
    print(f"  Final Quality (r_final): {r_final:.3f} (weight: {w_final})")
    print(f"  Improvement Gain (r_improve): {r_improve:.3f} (weight: {w_improve})")
    print(f"  Format Score: {format_score}")
    print(f"  Total Weighted Score: {total_score:.3f}")
    print("="*80 + "\n")

    return total_score


def compute_score_val_bleu(solution_str: str, 
                 ground_truth: str,
                 lg_pair: str,
                 use_reflex: bool = True,
                 format_reward: int = 1,
                 answer_reward: float = 1.0) -> float:
    """Computes BLEU score for model response during validation using Reflex-RL format.
    
    Args:
        solution_str: Raw model response string
        ground_truth: target ground truth data
        lg_pair: Language pair (e.g., 'zh-en')
        use_reflex: Whether to use Reflex-RL format extraction (default True)
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        BLEU score of the final translation
    """
    print("\n" + "="*80)
    print(" Processing Test Sample ".center(80, '='))
    
    solution_text = ground_truth
    
    # Extract draft, reflection, and final answer from Reflex-RL format
    draft_answer, reflection_text, final_answer, processed_str = extract_solution(solution_str)
    answer_text = final_answer if final_answer else draft_answer
    
    print(f"\n[Prompt + Response]\n{solution_str}")

    if answer_text:
        pred_status = compute_bleu(lg_pair, ground_truth, answer_text)
        print(f"Reference: {solution_text}")
        print(f"Hypothesis: {answer_text}")
        
        # 【优化】只在 draft 和 final 不同时才计算 draft BLEU
        if draft_answer and final_answer and draft_answer != final_answer:
            draft_bleu = compute_bleu(lg_pair, ground_truth, draft_answer)
            print(f"Draft: {draft_answer}")
            print(f"Draft BLEU: {draft_bleu:.2f}")
            print(f"Improvement: {pred_status - draft_bleu:.2f}")
    else:
        pred_status = compute_bleu(lg_pair, ground_truth, "")
        print(f"Reference: {solution_text}")
        print(f"Hypothesis: {processed_str}")
        
    total_score = pred_status
    print("\n" + "-"*80)
    print(f"BLEU Score: {total_score}")

    return total_score