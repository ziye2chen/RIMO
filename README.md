<h1 align="center">
    <br>RIMO
</h1>
<p align="center">
    <a href="http://huggingface.co/datasets/ziye2chen/ReIMO">
        <img alt="Static Badge" src="https://img.shields.io/badge/HuggingFace-ReIMO-yellow">
    </a>
    </a>
    <a href="https://github.com/ziye2chen/ReIMO">
        <img alt="Static Badge" src="https://img.shields.io/badge/Github-ReIMO-black">
    </a>
</p>

## 📚️ RIMO

![](img/math_model_scores.png)

RIMO (Remade International Mathematical Olympiad) is a purpose-built benchmark for probing large-language-model reasoning at true Olympiad level.

- 335 single-integer problems (RIMO-N) – each original IMO question is carefully rewritten so its answer is one unique integer, allowing 100 % deterministic grading.

- 472 full-proof problems (RIMO-P) – untouched IMO statements paired with vetted reference solutions and an open-source rubric grader for scalable proof assessment.

All 807 tasks are drawn exclusively from IMO contests and shortlists (1959 – 2023), covering the four canonical domains—algebra, geometry, number theory, combinatorics—and tagged by year, topic, and source. RIMO therefore delivers a high-resolution, noise-free yard-stick for evaluating both answer-finding and proof-writing skills in modern LLMs.

> [!NOTE]
>
> In the `RIMO` folder, we provide both `.jsonl` and `.csv` files for each task. The contents are the same. Just choose the one you prefer.

------

## 💡 News

- *2025-06-04*: We have released the RIMO dataset.

------

## 🔥 Test Your Model on RIMO-Main

RIMO-N is a set of 335 single-integer problems, each original IMO question is carefully rewritten so its answer is one unique integer, allowing 100 % deterministic grading. For the answer verification, we can directly compare the predicted integer with the correct answer.

### Your Own Model or Open-Source Model

`./code/RIMO_N_Open_Source.py` is a sample code for evaluating your own model or open-source model on RIMO-N. You can simply replace the `model` variable with your own model for inference.


```bash
cd ./code
python3 ./RIMO_N_Open_Source.py
```

### API

We also provide a sample code for evaluating the model with api. You can replace the `model_name` variable with your own model name, `api_key` with your own api key, and `base_url` with the url of the api.

```bash
cd ./code
python3 ./RIMO_N_API.py
```

------

## 🧩 Test Your Model on RIMO-P

RIMO-P is a set of 472 full-proof problems, untouched IMO statements paired with vetted reference solutions and an open-source rubric grader for scalable proof assessment. For the proof assessment, we can use the provided rubric to grade the proof and provide a score.

### Your Own Model or Open-Source Model

`./code/RIMO_P_Open_Source.py` is a sample code for evaluating your own model or open-source model on RIMO-P. You can simply replace the `model` variable with your own model for inference.


```bash
cd ./code
python3 ./RIMO_P_Open_Source.py
```

### Judge

We also provide a sample code for evaluating the answer with `Qwen/Qwen3-8B` judge. If you have your fine-tuned judge, you can replace the `MODEL_NAME` variable with your own judge name.

```bash
cd ./code
python3 ./RIMO_P_Judge.py
```

------

## 📎 Citation

If you use RIMO in your work, please cite the following paper:

<!-- ```
@article{chen2025reimo,
          title={ReIMO: A Remade International Mathematical Olympiad Benchmark for Evaluating Large Language Models},
          year={2025},
}
``` -->



