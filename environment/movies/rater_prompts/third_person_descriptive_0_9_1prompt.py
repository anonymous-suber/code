import os
from typing import List
from environment.LLM import LLMRater
from environment.LLM.llm import LLM
import re
from .third_person_descriptive_0_9 import ThirdPersonDescriptive09

from environment.memory import UserMovieInteraction
from environment.movies import Movie, MoviesLoader
from environment.users import User


class ThirdPersonDescriptive09_1Shot(ThirdPersonDescriptive09):
    def __init__(
        self,
        llm: LLM,
        current_items_features_list,
        previous_items_features_list,
        llm_render=False,
        llm_query_explanation=False,
        switch_user=False,
    ):
        super().__init__(
            llm,
            current_items_features_list,
            previous_items_features_list,
            llm_render,
            llm_query_explanation,
        )
        self.cache_few_shot_prompts = None
        self.switch_user = switch_user

    def _get_few_shot_prompts(self):
        if self.cache_few_shot_prompts is None:
            if self.switch_user:
                user = User(
                    "Nicholas",
                    "M",
                    26,
                    (
                        "a thrill-seeker who loves action and adventure and doesn't"
                        " care much for romance. He enjoys activities like rock"
                        " climbing, extreme sports, and exciting trips that make his"
                        " heart race. As a wilderness guide, he leads others through"
                        " tough terrains and shows them the beauty of wild places."
                        " During his free time, Nicholas loves reading gripping books"
                        " that take him on exciting journeys. He looks for friends who"
                        " also enjoy thrilling experiences and share his passion for"
                        " living life to the fullest. Adrenaline gives him an amazing"
                        " feeling, and he prefers action and adventure over romance any"
                        " day."
                    ),
                )
                base = os.path.dirname(os.path.abspath(__file__))
                items_loader = MoviesLoader(
                    os.path.join(base, "./sample_prompt_1.json")
                )
                movies = items_loader.load_items_from_ids([313369, 216015, 245891, 155])
                movie = movies[0]
                num_interacted = 0
                interactions = [
                    UserMovieInteraction(3, 0, 1),
                    UserMovieInteraction(9, 0, 1),
                    UserMovieInteraction(10, 0, 1),
                ]
                retrieved_items = movies[1:]
                prompt = self._get_prompt(
                    user,
                    movie,
                    num_interacted,
                    interactions,
                    retrieved_items,
                    do_rename=False,
                )
                explanation = (
                    "3 on a scale of 0 to 9, because Nicholas has a strong preference"
                    " for adrenaline-inducing action, thriller, and horror movies, he"
                    ' would likely rate the movie "La La Land" 3 out of 9. This is'
                    " evident from the description of Nicholas, which highlights his"
                    " enthusiasm for action-packed films that provide a surge of thrill"
                    " and excitement. In his previous film ratings, action movies"
                    " generally received higher scores, while films that didn't offer"
                    ' the same adrenaline rush, like "Fifty Shades of Grey", received'
                    ' lower ratings, such as a 2. As "La La Land" is a romantic musical'
                    " and not focused on action, it may not resonate as strongly with"
                    " Nicholas's taste for thrilling experiences. While the film is"
                    " generally well-liked with an average rating of 6.9, Nicholas's"
                    ' preference for adrenaline-filled plots might lead him to rate "La'
                    " La Land\" lower than the overall community rating. However, it's"
                    ' likely that he wouldn\'t rate it as low as "Fifty Shades of Grey"'
                    " due to its higher popularity and appreciation among viewers who"
                    " enjoy romance and musical genres."
                )
            else:
                user = User(
                    "Alex",
                    "M",
                    12,
                    (
                        "captivated by space exploration. With wide eyes and endless"
                        " wonder, he devours books on galaxies and dreams of becoming"
                        " an astronaut. Nights find him gazing at the stars, his"
                        " imagination soaring with each cosmic discovery. Alex's"
                        " passion for space knows no bounds as he reaches for the"
                        " stars."
                    ),
                )
                base = os.path.dirname(os.path.abspath(__file__))
                items_loader = MoviesLoader(
                    os.path.join(base, "./sample_prompt_1.json")
                )
                movies = items_loader.load_items_from_ids([269149, 953, 116977, 157336])
                movie = movies[0]
                num_interacted = 0
                interactions = [
                    UserMovieInteraction(7, 0, 1),
                    UserMovieInteraction(2, 0, 1),
                    UserMovieInteraction(10, 0, 1),
                ]
                retrieved_items = movies[1:]
                prompt = self._get_prompt(
                    user,
                    movie,
                    num_interacted,
                    interactions,
                    retrieved_items,
                    do_rename=False,
                )
                explanation = (
                    "8 on a scale of 0 to 9, because, even though the movie is not"
                    ' space-related, he previously enjoyed watching "Madagascar" and'
                    ' gave it a high rating. Since "Zootropolis" shares many'
                    ' similarities with "Madagascar," such as being animated movies'
                    " with animals living in a society resembling humans, it is highly"
                    ' likely that Alex also likes "Zootropolis." Both films explore'
                    " themes of friendship, identity, and the challenges of coexistence"
                    ' within diverse communities. Although "Zootropolis" doesn\'t have'
                    " a direct connection to space, Alex still appreciates it and gives"
                    " it a high rating. Furthermore, considering that children"
                    " generally have a fondness for animated movies, it is reasonable"
                    ' to assume that Alex would rate "Zootropolis" high." Additionally,'
                    ' "Zootropolis" is generally favored by a larger audience compared'
                    ' to "Madagascar," further supporting the idea that Alex would give'
                    " it a higher rating."
                )

            self.cache_few_shot_prompts = [
                {"role": "user", "content": prompt[0]["content"]},
                {
                    "role": "assistant",
                    "content": prompt[1]["content"] + explanation,
                },
            ]
        return self.cache_few_shot_prompts
