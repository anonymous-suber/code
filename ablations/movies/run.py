import argparse
import os

from environment.movies.movies_loader import MoviesLoader
from .src import (
    HighRatingStudy,
    LowRatingStudy,
    SamplingStudy,
    MovieSagasHistoryRandomStudy,
    GenrePreferencePaperStudy,
    SamplingSubsetInteractionsStudy,
)
import environment.LLM as LLM
from environment import Simulatio4RecSys
from environment.items_selection import GreedySelector
from environment.reward_shaping import IdentityRewardShaping
from environment.users import UsersLoader
from environment.movies.configs import (
    OPTIONS_LLM_RATER,
    OPTIONS_ITEMS_RETRIEVAL,
    get_enviroment_from_args,
    get_llm_rater,
    OPTIONS_REWARD_PERTURBATOR,
    get_reward_perturbator,
    get_items_retrieval,
    get_base_parser,
    get_user_dataset,
)


def parse_args():
    parser = get_base_parser()
    parser.add_argument("--skip-sampling", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    name = args.llm_model

    model = LLM.load_LLM(name)

    # During the ablation study we want to look at the raw ratings, not the reward for RL
    args.reward_shaping = "identity"

    def create_env(item: str, user_loader: UsersLoader):
        return Simulatio4RecSys(
            render_mode=None,
            items_loader=MoviesLoader(item),
            users_loader=user_loader,
            items_selector=GreedySelector(),
            reward_perturbator=get_reward_perturbator(args.perturbator, seed=42),
            items_retrieval=get_items_retrieval(args.items_retrieval),
            llm_rater=get_llm_rater(
                args.llm_rater, model, args.items_retrieval != "none"
            ),
            reward_shaping=IdentityRewardShaping(seed=42),
        )

    name_report = name.replace("/", "_")
    exp_name = (
        f"{name_report}-{args.llm_rater}-{args.perturbator}-{args.items_retrieval}"
    )
    if args.user_dataset == "basic":
        exp_name += "-basic_users"
    elif args.user_dataset == "sampled_genres":
        exp_name += "-sampled_genres"

    if args.debug:
        exp_name += "-DEBUG"

    genre_study = GenrePreferencePaperStudy(create_env, exp_name)
    genre_study.run()

    high_study = HighRatingStudy(create_env, exp_name)
    high_study.run()

    low_study = LowRatingStudy(create_env, exp_name)
    low_study.run()

    movie_sagas_history_study_random = MovieSagasHistoryRandomStudy(
        create_env, exp_name, user_dataset=get_user_dataset(args.user_dataset)
    )
    movie_sagas_history_study_random.run()

    if not args.skip_sampling:
        sampling_study = SamplingStudy(
            lambda: get_enviroment_from_args(model, args), exp_name
        )
        sampling_study.run()
