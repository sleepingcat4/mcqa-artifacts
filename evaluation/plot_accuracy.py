import datasets
import argparse
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))

from data_loader import create_data_evaluation, PromptType, DatasetName, ModelName

EXPERIMENTS_ARTIFACT = [PromptType.normal, PromptType.artifact_choices]
EXPERIMENTS_MEMORIZE = [
    PromptType.normal, PromptType.memorization_empty,
    PromptType.memorization_gold, PromptType.memorization_no_choice
]
EXPERIMENTS_INFER_QUESTION = EXPERIMENTS_ARTIFACT + [
    PromptType.artifact_choices_cot_twostep_generated, PromptType.artifact_choices_cot_twostep_random
]


def setup():
    def enum_type(enum):
        enum_members = {e.name: e for e in enum}

        def converter(input):
            out = []
            for x in input.split():
                if x in enum_members:
                    out.append(enum_members[x])
                else:
                    raise argparse.ArgumentTypeError(
                        f"You used {x}, but value must be one of {', '.join(enum_members.keys())}"
                    )
            return out

        return converter

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        '-m',
        type=enum_type(ModelName),
        help="(Nick)name of models",
        default=[ModelName.llama_7b, ModelName.falcon_40b, ModelName.mistral_7b, ModelName.phi_2]
    )
    parser.add_argument(
        "--datasets",
        type=enum_type(DatasetName),
        help="Name of the dataset (in dataset_name column)",
        default=[DatasetName.ARC, DatasetName.mmlu, DatasetName.HellaSwag]
    )
    parser.add_argument(
        '--experiments',
        type=str,
        help='experiments to run',
        default="artifact",
        choices=["artifact", "memorization", "inference"]
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        help="Absolute directory of the output results folder",
        default="/mcqa-artifacts/results"
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        help="Absolute directory of folder for saving plots",
        default="/mcqa-artifacts/images"
    )

    args = parser.parse_args()

    args.models = [model.value for model in args.models]

    if args.experiments == "artifact":
        args.experiments = EXPERIMENTS_ARTIFACT
    elif args.experiments == "memorization":
        args.experiments = EXPERIMENTS_MEMORIZE
    elif args.experiments == "inference":
        args.experiments = EXPERIMENTS_INFER_QUESTION

    print("models:", args.models)
    print("experiments:", [experiment.value for experiment in args.experiments])
    print("datasets:", [dataset.value for dataset in args.datasets])

    return args


t_test_strats = {}

colors = {
    'Blue': '#4565ae',
    'Red': '#cd2428',
    'Light Red': '#faa5a7',
    'Dark Red': '#910306',
    'Light Purple': '#cc9bfa',
    'Dark Purple': '#552482'
}

colors_map = {
    PromptType.normal.value: colors['Blue'],
    PromptType.artifact_choices.value: colors['Red'],
    PromptType.memorization_no_choice.value: colors['Light Red'],
    PromptType.memorization_empty.value: colors['Red'],
    PromptType.memorization_gold.value: colors['Dark Red'],
    PromptType.artifact_choices_cot_twostep_generated.value: colors['Light Purple'],
    PromptType.artifact_choices_cot_twostep_random.value: colors['Dark Purple'],
    PromptType.artifact_choices_cot.value: colors['Dark Purple'],
}

pt_names_map = {
    PromptType.normal.value: 'Full Prompt',
    PromptType.artifact_choices.value: 'Choices-Only Prompt',
    PromptType.memorization_empty.value: 'Empty Choices',
    PromptType.memorization_gold.value: 'Gold Choices',
    PromptType.memorization_no_choice.value: 'No Choices',
    PromptType.artifact_choices_cot_twostep_generated.value: 'Infer the Question 2-step',
    PromptType.artifact_choices_cot_twostep_random.value: 'Random Question Prompt',
    PromptType.artifact_choices_cot.value: 'Infer the Question 1-step',
}

model_names_map = {
    'llama_70b': 'LLaMA 70B',
    'llama_13b': 'LLaMA 13B',
    'llama_7b': 'LLaMA 7B',
    'falcon_40b': 'Falcon 40B',
    'mistral_7b': 'Mixtral 8x7B',
    'phi_2': 'Phi 2'
}

patterns = {
    PromptType.artifact_choices_cot_twostep_generated.value: '///',
}


reported_res = {
    'llama_7b': {
        DatasetName.ARC: 0.5307, DatasetName.HellaSwag: 0.7859, DatasetName.mmlu: 0.3876, DatasetName.Winogrande: 0.7403
    },
    'llama_13b': {
        DatasetName.ARC: 0.5939, DatasetName.HellaSwag: 0.8213, DatasetName.mmlu: 0.5577, DatasetName.Winogrande: 0.7664
    },
    'llama_70b': {
        DatasetName.ARC: 0.6732, DatasetName.HellaSwag: 0.8733, DatasetName.mmlu: 0.6983, DatasetName.Winogrande: 0.8374
    }
}


def format_models(models):
    models_ = []
    for m in models:
        if 'llama' in m:
            models_.append(f'LLaMA {m.split()[1].upper()}')
        elif 'falcon' in m:
            models_.append(f'Falcon {m.split()[1].upper()}')
        elif 'gpt' in m:
            models_.append(f'GPT {m.split()[1].upper()}')
        else:
            models_.append(m)
    return models_


def format_dataset(ds):
    if ds == DatasetName.mmlu:
        return 'MMLU (5-shot)'
    elif ds == DatasetName.ARC:
        return 'ARC (25-shot)'
    elif ds == DatasetName.HellaSwag:
        return 'HellaSwag (10-shot)'
    else:
        return ds.value


def convert_raw_text(rt):
    if rt is None:
        return 'Z'
    # rt_old = rt   # unused
    if 'Answer:' in rt:
        rt = rt[rt.index('Answer:') + len('Answer:'):]
    rt = rt[:4]
    rt = rt.strip()
    if rt in {'(A)', '(1)'}:
        return 'A'
    if rt in {'(B)', '(2)'}:
        return 'B'
    if rt in {'(3)', '(C)'}:
        return 'C'
    if rt in {'(D)', '(4)'}:
        return 'D'
    return 'Z'


def get_llm_answer(prompt, answer, choices_true):
    prompt = prompt.split('\n\n')[-1]    
    choices_txt = prompt[prompt.index('Choices:\n') + len('Choices:\n'):prompt.index('Answer:')]
    choices = [x[3:].strip() for x in choices_txt.split('\n')[:-1]]
    if ord(answer) - ord('A') >= len(choices):
        return ''
    return choices[ord(answer) - ord('A')]


def compute_accuracy(p, t):
    arr = []
    for i in range(len(p)):
        p_, t_ = p[i], t[i]
        if p_ == ord('Z'):
            arr.append(0.25)
        else:
            arr.append(int(p_ == t_))
    return np.mean(arr), arr


if __name__ == '__main__':
    args = setup()
    ds = datasets.load_dataset('nbalepur/mcqa_artifacts')

    fig, axs_ = plt.subplots(1, 3, figsize=(14, 3))
    bar_width = 0.3 if len(args.experiments) == 2 else 0.2
    legend_margin = 0.22  # higher is more space
    p_value_cutoff = 0.00005
    try:
        axs = list(axs_.ravel())
    except:
        axs = [axs_]
        axs_ = [[axs_]]
    idx_ = 0

    random_guess_accuracy, majority_accuracy = dict(), dict()

    for dataset_idx, dataset in enumerate(args.datasets):
        benchmark_graph_data = {'LLM': [], 'Strategy': [], 'Accuracy': []}
        arr_map = dict()

        for model_nickname in args.models:
            for pt in args.experiments:
                data = create_data_evaluation(ds, dataset, pt)
                questions, choices, answer_letters, answer_texts = data['questions'], data['choices'], data['answer_letters'], data['answer_texts']

                if dataset not in majority_accuracy:
                    freq = dict()
                    for a in answer_letters:
                        freq[a] = freq.get(a, 0) + 1
                    v = list(freq.values())
                    max_item = max(freq.items(), key=lambda item: item[1])[0]
                    majority_arr_ = [max_item for _ in range(len(questions))]
                    majority_arr = [int(majority_arr_[m_idx] == answer_letters[m_idx]) for m_idx in range(len(majority_arr_))]
                    majority_accuracy[dataset] = max(v) / sum(v)

                res_dir = f'{args.res_dir}/{dataset.value}/{model_nickname}/{pt.value}.pkl'
                with open(res_dir, 'rb') as handle:
                    res = pickle.load(handle)

                pred_answer_letters = [convert_raw_text(rt) for rt in res['raw_text']]
                orig_pred, orig_true = len(pred_answer_letters), len(answer_letters)
                if PromptType.artifact_choices_cot_twostep_generated in args.experiments and dataset == DatasetName.mmlu:
                    valid_idxs = set(range(int(0.75 * orig_true)))
                else:
                    valid_idxs = set(range(orig_true))

                pred_answer_letters = [pred_answer_letters[idx] for idx in range(len(pred_answer_letters))]

                pred_idx = [ord(letter) for letter in pred_answer_letters]
                true_idx = [ord(letter) for letter in answer_letters]

                true_idx = [true_idx[idx] for idx in valid_idxs]
                pred_idx = [pred_idx[idx] for idx in valid_idxs]

                assert len(pred_idx) == len(true_idx)

                accuracy, arr = compute_accuracy(pred_idx, true_idx)
                arr_map[(model_nickname, pt.value)] = arr

                benchmark_graph_data['Accuracy'].append(accuracy)
                benchmark_graph_data['LLM'].append(model_names_map[model_nickname])
                benchmark_graph_data['Strategy'].append(pt.value)

        df = pd.DataFrame(benchmark_graph_data)
        df['Strategy'] = pd.Categorical(df['Strategy'], categories=[e.value for e in args.experiments], ordered=True) #+ ['reported']
        df['LLM'] = pd.Categorical(df['LLM'], categories=[model_names_map[m] for m in args.models], ordered=True)

        ax = axs[idx_]
        idx_ += 1

        offset = 0

        llm_positions = list(range(len(args.models)))
        relative_positions = list(0 + np.arange(len(args.experiments) + 1))

        # ax.axhline(random_guess_accuracy[dataset], color='orange', linewidth=2, label='Guessing', ls='--')
        ax.axhline(majority_accuracy[dataset], color='orange', linewidth=2, label='Majority Class', ls='--')

        for idx, strategy in enumerate(df['Strategy'].unique()):
            accuracies = df[df['Strategy'] == strategy]['Accuracy'].values
            bars = ax.bar(
                [pos + relative_positions[idx]*bar_width for pos in llm_positions],
                accuracies,
                bar_width,
                label=pt_names_map[strategy],
                color=colors_map[strategy],
                hatch=patterns.get(strategy, '')
            )

            if strategy in t_test_strats:
                for bar_idx in range(len(bars)):
                    bar_offset = 0.02
                    t_test = stats.ttest_ind(arr_map[(args.models[bar_idx], strategy)], majority_arr, alternative='greater')

                    if t_test.pvalue < p_value_cutoff:
                        ax.text(
                            bars[bar_idx].xy[0] + 0.5 * bar_width, bars[bar_idx].xy[1] + bars[bar_idx]._height + bar_offset,
                            '*',
                            fontsize=12,
                            fontweight='semibold',
                            verticalalignment='center',
                            horizontalalignment='center'
                        )

        ax.set_title(f'{format_dataset(dataset)}')
        ax.set_xticks(np.array(llm_positions) + (len(args.experiments) * 0.5) * bar_width - 0.5 * bar_width)
        ax.set_xticklabels([m for m in df['LLM'].unique()])
        ax.set_ylim(top=1.0)
        ax.set_ylim(bottom=0)

        if dataset_idx % 2 == 0:
            ax.set_ylabel('Accuracy')

    all_handles = []
    all_labels = []

    if isinstance(axs_[0], list):
        for ax in axs_:
            for a in ax:
                h, l = a.get_legend_handles_labels()
                all_handles.extend(h)
                all_labels.extend(l)
    else:
        for a in axs_:
            h, l = a.get_legend_handles_labels()
            all_handles.extend(h)
            all_labels.extend(l)

    # To ensure that the legend is unique
    unique = [(h, l) for i, (h, l) in enumerate(zip(all_handles, all_labels)) if l not in all_labels[:i]]
    unique_handles, unique_labels = zip(*unique)

    fig.legend(unique_handles, unique_labels, fontsize=12, loc='lower center', ncol=5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=legend_margin)
    print(f"Saving accuracy plot to {args.plot_dir}/plot_accuracy.png")
    plt.savefig(os.path.join(args.plot_dir, "plot_accuracy.png"), dpi=500)
