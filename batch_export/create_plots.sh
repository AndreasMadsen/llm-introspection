python export/plot_classify_y-accuracy_x-prompt.py --model-name llama2-70b --system-message none --datasets IMDB bAbI-1 MCTest RTE --split test
#python export/plot_redacted_y-faithfulness_x-prompt.py --model-name llama2-70b --system-message none --datasets IMDB bAbI-1 MCTest RTE --split test
#python export/plot_importance_y-faithfulness_x-prompt.py --model-name llama2-70b --system-message none --datasets IMDB bAbI-1 MCTest RTE --split test
#python export/plot_counterfactual_y-faithfulness_x-prompt.py --model-name llama2-70b --system-message none --datasets IMDB bAbI-1 MCTest RTE --split test
python export/plot_explain_f-task_y-faithfulness_x-prompt.py --model-name llama2-70b --system-message none --datasets IMDB bAbI-1 MCTest RTE --split test

python export/plot_classify_y-accuracy_x-model.py --datasets IMDB bAbI-1 MCTest RTE --split test
#python export/plot_explain_y-faithfulness_x-model.py --task counterfactual --datasets IMDB bAbI-1 MCTest RTE --split test
#python export/plot_explain_y-faithfulness_x-model.py --task importance --datasets IMDB bAbI-1 MCTest RTE --split test
#python export/plot_explain_y-faithfulness_x-model.py --task redacted --datasets IMDB bAbI-1 MCTest RTE --split test
python export/plot_explain_f-task_y-faithfulness_x-model.py --datasets IMDB bAbI-1 MCTest RTE --split test

python export/plot_explain_f-task_y-faithfulness_x-model.py --datasets IMDB bAbI-1 RTE --split test --format website
