python export/plot_classify_accuracy.py --model-name llama2-70b --system-message none --datasets IMDB bAbI-1 MCTest RTE --split train
python export/plot_redacted_faithfulness.py --model-name llama2-70b --system-message none --datasets IMDB bAbI-1 MCTest RTE --split train
python export/plot_importance_faithfulness.py --model-name llama2-70b --system-message none --datasets IMDB bAbI-1 MCTest RTE --split train
python export/plot_counterfactual_faithfulness.py --model-name llama2-70b --system-message none --datasets IMDB bAbI-1 MCTest RTE --split train
