# Reflex-RL: Self-Correcting Machine Translation via Reflective Reinforcement Learning

This document describes the modifications made to the MT-R1-Zero framework to implement Reflex-RL, a novel approach for machine translation that mimics human translator workflow through a "draft-reflection-refinement" process.

## Overview

Reflex-RL transforms the traditional single-step translation process into a structured three-stage generation:

1. **Draft Generation**: Model produces an initial translation
2. **Reflective Critique**: Model analyzes the draft and identifies potential issues  
3. **Final Translation**: Model generates a refined translation based on the reflection

## Key Modifications

### 1. Data Processing (`data/process_data.py`)

**Added**: New template type `'reflex'` that instructs the model to use the three-stage format:

```python
elif template_type == 'reflex':
    prefix = f"""A conversation between User and Assistant. The User asks for a translation from {src_lang_name} to {tgt_lang_name}, and the Assistant follows a reflective translation process. The Assistant first provides a draft translation, then reflects critically on the draft to identify potential issues, and finally provides a refined translation based on the reflection. The process follows this format: <draft>initial translation</draft><reflection>critical analysis of the draft</reflection><final>refined translation</final>. \n\nUser:{user_input}\nAssistant:"""
```

### 2. Reward Function (`verl/utils/reward_score/mt_score.py`)

**Major Rewrite**: Implemented two-dimensional reward mechanism:

#### A. Enhanced Output Parsing
- `extract_solution()`: Now extracts `draft`, `reflection`, and `final` components
- `validate_response_structure()`: Validates new three-stage tag format

#### B. Two-Dimensional Reward Components

1. **Final Quality Reward (`r_final`)**
   - Evaluates absolute quality of the final translation
   - Uses COMET/BLEU scores as in original framework

2. **Improvement Gain Reward (`r_improve`)** - **Core Innovation**
   - Directly rewards the improvement from draft to final translation
   - `r_improve = COMET(final) - COMET(draft)`
   - Creates causal link between reflection quality and translation improvement

#### C. Weighted Total Reward
```python
R_total = w_final * r_final + w_improve * r_improve
```

### 3. Training Configuration (`verl/trainer/config/ppo_trainer_mt.yaml`)

**Added** Reflex-RL specific parameters:

```yaml
algorithm:
  # Existing parameters...
  
  # Reflex-RL specific parameters
  use_reflex: True      # Enable three-stage generation
  w_final: 1.0          # Weight for final quality reward
  w_improve: 2.0        # Weight for improvement gain reward (emphasized)
```

### 4. Reward Manager (`verl/trainer/reward_manager.py`)

**Modified**: 
- Added support for new reward parameters (`use_reflex`, `w_final`, `w_improve`)
- Updated function calls to pass these parameters to scoring functions

### 5. Training Script (`main_grpo.sh`)

**Changed**: Template type from `"base"` to `"reflex"` for data processing

## Usage

### Training with Reflex-RL

1. **Data Processing**:
```bash
python3 data/process_data.py \
    --template_type "reflex" \
    --train_files ... \
    --test_files ...
```

2. **Training**:
```bash
bash main_grpo.sh
```

The training script automatically uses the reflex configuration.

### Configuration Options

Key parameters in `ppo_trainer_mt.yaml`:

- `use_reflex: True/False` - Enable/disable Reflex-RL mode
- `w_final: float` - Weight for final translation quality
- `w_improve: float` - Weight for improvement reward (typically highest)

### Expected Output Format

With Reflex-RL enabled, the model should generate:

```
<draft>Initial translation with potential issues</draft>
<reflection>Critical analysis: This translation has issues with X, Y, and Z...</reflection>
<final>Improved translation addressing the identified issues</final>
```

## Benefits

1. **Explicit Self-Correction**: Model learns to identify and fix its own mistakes
2. **Interpretability**: Reflection text provides insight into the translation process  
3. **Better Training Signal**: Improvement reward directly incentivizes effective correction
4. **Human-like Workflow**: Mimics professional translator methodology

## Compatibility

The implementation maintains backward compatibility:
- Legacy format still supported when `use_reflex: False`
- Existing evaluation scripts work with both formats
- Original MT-R1-Zero functionality preserved

## Key Research Contributions

1. **Novel Reward Design**: First RL framework to directly reward "improvement" behavior
2. **Self-Correction Learning**: Teaches LLMs to internalize revision capabilities
3. **Process-Oriented Training**: Optimizes not just outcomes but the reasoning process

This implementation provides a foundation for exploring reflective learning in machine translation and can be extended to other text generation tasks.
