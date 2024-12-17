# Test Structure
```
.
├── calib_tools.py
├── categories.py
├── crop.py
├── data
│   ├── auxiliary_train
│   │   ├── arc_easy.csv
│   │   ├── arc_hard.csv
│   │   ├── aux_law_90s.csv
│   │   ├── mc_test.csv
│   │   ├── obqa.csv
│   │   ├── race.csv
│   │   ├── science_elementary.csv
│   │   └── science_middle.csv
│   ├── dev
│   │   ├── abstract_algebra_dev.csv
│   │   ├── anatomy_dev.csv
│   │   ├── astronomy_dev.csv
│   │   ├── business_ethics_dev.csv
│   │   ├── clinical_knowledge_dev.csv
│   │   ├── college_biology_dev.csv
│   │   ├── college_chemistry_dev.csv
│   │   ├── college_computer_science_dev.csv
│   │   ├── college_mathematics_dev.csv
│   │   ├── college_medicine_dev.csv
│   │   ├── college_physics_dev.csv
│   │   ├── computer_security_dev.csv
│   │   ├── conceptual_physics_dev.csv
│   │   ├── econometrics_dev.csv
│   │   ├── electrical_engineering_dev.csv
│   │   ├── elementary_mathematics_dev.csv
│   │   ├── formal_logic_dev.csv
│   │   ├── global_facts_dev.csv
│   │   ├── high_school_biology_dev.csv
│   │   ├── high_school_chemistry_dev.csv
│   │   ├── high_school_computer_science_dev.csv
│   │   ├── high_school_european_history_dev.csv
│   │   ├── high_school_geography_dev.csv
│   │   ├── high_school_government_and_politics_dev.csv
│   │   ├── high_school_macroeconomics_dev.csv
│   │   ├── high_school_mathematics_dev.csv
│   │   ├── high_school_microeconomics_dev.csv
│   │   ├── high_school_physics_dev.csv
│   │   ├── high_school_psychology_dev.csv
│   │   ├── high_school_statistics_dev.csv
│   │   ├── high_school_us_history_dev.csv
│   │   ├── high_school_world_history_dev.csv
│   │   ├── human_aging_dev.csv
│   │   ├── human_sexuality_dev.csv
│   │   ├── international_law_dev.csv
│   │   ├── jurisprudence_dev.csv
│   │   ├── logical_fallacies_dev.csv
│   │   ├── machine_learning_dev.csv
│   │   ├── management_dev.csv
│   │   ├── marketing_dev.csv
│   │   ├── medical_genetics_dev.csv
│   │   ├── miscellaneous_dev.csv
│   │   ├── moral_disputes_dev.csv
│   │   ├── moral_scenarios_dev.csv
│   │   ├── nutrition_dev.csv
│   │   ├── philosophy_dev.csv
│   │   ├── prehistory_dev.csv
│   │   ├── professional_accounting_dev.csv
│   │   ├── professional_law_dev.csv
│   │   ├── professional_medicine_dev.csv
│   │   ├── professional_psychology_dev.csv
│   │   ├── public_relations_dev.csv
│   │   ├── security_studies_dev.csv
│   │   ├── sociology_dev.csv
│   │   ├── us_foreign_policy_dev.csv
│   │   ├── virology_dev.csv
│   │   └── world_religions_dev.csv
│   ├── possibly_contaminated_urls.txt
│   ├── README.txt
│   ├── test
│   │   ├── abstract_algebra_test.csv
│   │   ├── anatomy_test.csv
│   │   ├── astronomy_test.csv
│   │   ├── business_ethics_test.csv
│   │   ├── clinical_knowledge_test.csv
│   │   ├── college_biology_test.csv
│   │   ├── college_chemistry_test.csv
│   │   ├── college_computer_science_test.csv
│   │   ├── college_mathematics_test.csv
│   │   ├── college_medicine_test.csv
│   │   ├── college_physics_test.csv
│   │   ├── computer_security_test.csv
│   │   ├── conceptual_physics_test.csv
│   │   ├── econometrics_test.csv
│   │   ├── electrical_engineering_test.csv
│   │   ├── elementary_mathematics_test.csv
│   │   ├── formal_logic_test.csv
│   │   ├── global_facts_test.csv
│   │   ├── high_school_biology_test.csv
│   │   ├── high_school_chemistry_test.csv
│   │   ├── high_school_computer_science_test.csv
│   │   ├── high_school_european_history_test.csv
│   │   ├── high_school_geography_test.csv
│   │   ├── high_school_government_and_politics_test.csv
│   │   ├── high_school_macroeconomics_test.csv
│   │   ├── high_school_mathematics_test.csv
│   │   ├── high_school_microeconomics_test.csv
│   │   ├── high_school_physics_test.csv
│   │   ├── high_school_psychology_test.csv
│   │   ├── high_school_statistics_test.csv
│   │   ├── high_school_us_history_test.csv
│   │   ├── high_school_world_history_test.csv
│   │   ├── human_aging_test.csv
│   │   ├── human_sexuality_test.csv
│   │   ├── international_law_test.csv
│   │   ├── jurisprudence_test.csv
│   │   ├── logical_fallacies_test.csv
│   │   ├── machine_learning_test.csv
│   │   ├── management_test.csv
│   │   ├── marketing_test.csv
│   │   ├── medical_genetics_test.csv
│   │   ├── miscellaneous_test.csv
│   │   ├── moral_disputes_test.csv
│   │   ├── moral_scenarios_test.csv
│   │   ├── nutrition_test.csv
│   │   ├── philosophy_test.csv
│   │   ├── prehistory_test.csv
│   │   ├── professional_accounting_test.csv
│   │   ├── professional_law_test.csv
│   │   ├── professional_medicine_test.csv
│   │   ├── professional_psychology_test.csv
│   │   ├── public_relations_test.csv
│   │   ├── security_studies_test.csv
│   │   ├── sociology_test.csv
│   │   ├── us_foreign_policy_test.csv
│   │   ├── virology_test.csv
│   │   └── world_religions_test.csv
│   └── val
│       ├── abstract_algebra_val.csv
│       ├── anatomy_val.csv
│       ├── astronomy_val.csv
│       ├── business_ethics_val.csv
│       ├── clinical_knowledge_val.csv
│       ├── college_biology_val.csv
│       ├── college_chemistry_val.csv
│       ├── college_computer_science_val.csv
│       ├── college_mathematics_val.csv
│       ├── college_medicine_val.csv
│       ├── college_physics_val.csv
│       ├── computer_security_val.csv
│       ├── conceptual_physics_val.csv
│       ├── econometrics_val.csv
│       ├── electrical_engineering_val.csv
│       ├── elementary_mathematics_val.csv
│       ├── formal_logic_val.csv
│       ├── global_facts_val.csv
│       ├── high_school_biology_val.csv
│       ├── high_school_chemistry_val.csv
│       ├── high_school_computer_science_val.csv
│       ├── high_school_european_history_val.csv
│       ├── high_school_geography_val.csv
│       ├── high_school_government_and_politics_val.csv
│       ├── high_school_macroeconomics_val.csv
│       ├── high_school_mathematics_val.csv
│       ├── high_school_microeconomics_val.csv
│       ├── high_school_physics_val.csv
│       ├── high_school_psychology_val.csv
│       ├── high_school_statistics_val.csv
│       ├── high_school_us_history_val.csv
│       ├── high_school_world_history_val.csv
│       ├── human_aging_val.csv
│       ├── human_sexuality_val.csv
│       ├── international_law_val.csv
│       ├── jurisprudence_val.csv
│       ├── logical_fallacies_val.csv
│       ├── machine_learning_val.csv
│       ├── management_val.csv
│       ├── marketing_val.csv
│       ├── medical_genetics_val.csv
│       ├── miscellaneous_val.csv
│       ├── moral_disputes_val.csv
│       ├── moral_scenarios_val.csv
│       ├── nutrition_val.csv
│       ├── philosophy_val.csv
│       ├── prehistory_val.csv
│       ├── professional_accounting_val.csv
│       ├── professional_law_val.csv
│       ├── professional_medicine_val.csv
│       ├── professional_psychology_val.csv
│       ├── public_relations_val.csv
│       ├── security_studies_val.csv
│       ├── sociology_val.csv
│       ├── us_foreign_policy_val.csv
│       ├── virology_val.csv
│       └── world_religions_val.csv
├── data.tar.gz
├── eval_gsm8k.py
├── eval_humaneval.py
├── eval_mmlu.py
├── evaluate_flan.py
├── evaluate_mmlu.py
├── evaluate.py
├── gsm8k_prompt.txt
├── gsm8k_results.jsonl
├── gsm8k_test.jsonl
├── human-eval
│   ├── data
│   │   ├── example_problem.jsonl
│   │   ├── example_samples.jsonl
│   │   ├── HumanEval.jsonl
│   │   ├── HumanEval.jsonl.gz
│   │   ├── HumanEval_res.jsonl
│   │   ├── HumanEval_res.jsonl_results.jsonl
│   │   ├── HumanEval_res_origin.jsonl
│   │   └── HumanEval_res_origin.jsonl_results.jsonl
│   ├── human_eval
│   │   ├── data.py
│   │   ├── evaluate_functional_correctness.py
│   │   ├── evaluation.py
│   │   ├── execution.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── data.cpython-38.pyc
│   │       ├── evaluate_functional_correctness.cpython-38.pyc
│   │       ├── evaluation.cpython-38.pyc
│   │       ├── execution.cpython-38.pyc
│   │       └── __init__.cpython-38.pyc
│   ├── human_eval.egg-info
│   │   ├── dependency_links.txt
│   │   ├── entry_points.txt
│   │   ├── PKG-INFO
│   │   ├── requires.txt
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── LICENSE
│   ├── README.md
│   ├── requirements.txt
│   └── setup.py
├── LICENSE
├── logger.py
├── log.txt
├── __pycache__
│   ├── categories.cpython-312.pyc
│   ├── categories.cpython-38.pyc
│   └── logger.cpython-38.pyc
├── README.md
├── results
│   └── results_Llama
│       ├── abstract_algebra.csv
│       ├── anatomy.csv
│       ├── astronomy.csv
│       ├── business_ethics.csv
│       ├── clinical_knowledge.csv
│       ├── college_biology.csv
│       ├── college_chemistry.csv
│       ├── college_computer_science.csv
│       ├── college_mathematics.csv
│       ├── college_medicine.csv
│       ├── college_physics.csv
│       ├── computer_security.csv
│       ├── conceptual_physics.csv
│       ├── econometrics.csv
│       ├── electrical_engineering.csv
│       ├── elementary_mathematics.csv
│       ├── formal_logic.csv
│       ├── global_facts.csv
│       ├── high_school_biology.csv
│       ├── high_school_chemistry.csv
│       ├── high_school_computer_science.csv
│       ├── high_school_european_history.csv
│       ├── high_school_geography.csv
│       ├── high_school_government_and_politics.csv
│       ├── high_school_macroeconomics.csv
│       ├── high_school_mathematics.csv
│       ├── high_school_microeconomics.csv
│       ├── high_school_physics.csv
│       ├── high_school_psychology.csv
│       ├── high_school_statistics.csv
│       ├── high_school_us_history.csv
│       ├── high_school_world_history.csv
│       ├── human_aging.csv
│       ├── human_sexuality.csv
│       ├── international_law.csv
│       ├── jurisprudence.csv
│       ├── logical_fallacies.csv
│       ├── machine_learning.csv
│       ├── management.csv
│       ├── marketing.csv
│       ├── medical_genetics.csv
│       ├── miscellaneous.csv
│       ├── moral_disputes.csv
│       ├── moral_scenarios.csv
│       ├── nutrition.csv
│       ├── philosophy.csv
│       ├── prehistory.csv
│       ├── professional_accounting.csv
│       ├── professional_law.csv
│       ├── professional_medicine.csv
│       ├── professional_psychology.csv
│       ├── public_relations.csv
│       ├── security_studies.csv
│       ├── sociology.csv
│       ├── us_foreign_policy.csv
│       ├── virology.csv
│       └── world_religions.csv
└── test_calibration.py
```