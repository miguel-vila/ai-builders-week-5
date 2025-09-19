# AI Builders Week 5

## Objective

To fine-tune a model to receive a letterbox review and predict the star rating,
using this [dataset](https://www.kaggle.com/datasets/riyosha/letterboxd-movie-reviews-90000).

## Data preparation

In [./explore-data.ipynb](./explore-data.ipynb) I prepped the data:

- filtered out non-english reviews
- filtered in reviews for movies that were tagged as "watched".
- checked the distribution of ratings
- sampled 200 reviews per rating (there are less than 300 ratings with half a star, so in order to keep things distributed I set a max of 200).
- split into training (80%), validation (10%) and test (10%) sets.

Here's some sample data:

```
Samples for rating ★★★★★:
 Review: First time on a big screen and I'm literally overwhelmed: after almost thirty years Satoshi Kon is still beyond the beyond. - Movie: perfect-blue
 Review: That thing going around on TikTok that’s like “what’s an acting performance that was SO GOOD that you forgot it was acting??” and it’s just this in its entirety - Movie: whos-afraid-of-virginia-woolf
 Review: An advertisement of silence from god in a land of souls without eyelids. - Movie: werckmeister-harmonies
 Review: bob fosse was a human bojack horseman - Movie: all-that-jazz
 Review: About to lose my fucking minddddddddd - Movie: opening-night


Samples for rating ★★½:
 Review: Halfway through this I hit the play button on accident which caused WAP to start playing and honestly? Heightened the experience. - Movie: lawrence-of-arabia
 Review: "I had to execute someone, and there is something about it I didn't like." -Lawrence, Four hours of colonialism, sand, brown face and boredom.. not my thing but the cinematography is outstanding. - Movie: lawrence-of-arabia
 Review: Before watching I always assumed this was some crazy mystery thriller, no idea why. Instead it’s a relatively boring drama about a dickhead. - Movie: paris-texas
 Review: Definitely not for me. I had to watch a dude walk back and forth across an empty pool for what felt like 30 minutes. This is what you’d call an art piece. - Movie: nostalgia-1983
 Review: Wait... you're telling me the movie that got people so excited this year is a 22x-longer version of this??Oh, boy... - Movie: marcel-the-shell-with-shoes-on


Samples for rating ½:
 Review: Qui Gon Jinn is not this old. 0/10 - Movie: oldboy
 Review: i have a rhetorical question-what if that tampon didnt fall out of jojo siwas punani😔(1/10) - Movie: the-fifth-seal
 Review: what the fuck was that - Movie: war-and-peace
 Review: Unbelievably shitty. Shit characters and doesn't even star Dunkaccino - Movie: i-am-cuba
 Review: the only good part of this movie was rei farming - Movie: evangelion-3010-thrice-upon-a-time
 ```

## Target measures

I started with [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy)
as a target measure to look into the test set performance. But I think this might
not be correct as this doesn't take into account whether the predicted rating
is close numerically to the true rating. In the different variants that I ran
(with different hyperparameters, models, unfrozen layers and other things) I didn't get an
accuracy of more than `0.35`, being around 3 times better than random guessing
(which would be `0.1` for 10 different ratings)

Asking ChatGPT it told me to use other metrics like [Mean Absolute Error (MAE)](https://en.wikipedia.org/wiki/Mean_absolute_error) and [Cohen's Kappa (QWK)](https://numiqo.com/tutorial/cohens-kappa) to get a better sense of the model's performance.

For this task, it seems good to aim for a MAE of less than `1.0` (meaning that on average the predicted rating is within 1 star of the true rating) and a QWK of more than `0.5` (indicating some agreement between predicted and true ratings).

## How to run

### Download dataset

```bash
uv run kaggle datasets download -d riyosha/letterboxd-movie-reviews-90000 -p data
unzip ./data/letterboxd-movie-reviews-90000.zip -d ./data
```

### Baseline

The [./baseline.ipynb](./baseline.ipynb) notebook computes some baseline metrics
against the test set using an unrefined `bert-base-uncased` model:

- QWK: `0.0736`
- MAE: `1.32`
- Accuracy: `0.1`

### Training and evaluation: classification

At first I tried fine-tuning bert models using `AutoModelForSequenceClassification`.
You can see the code in [./fine-tune-bert.ipynb](./fine-tune-bert.ipynb).
I don't think these results were very good, I tried the base and large models,
as well as different learning rates and number of epochs.

The best accuracy I got was `0.31` (again, accuracy being not a good metric here),
by unfreezing the 11th layer as well and with a batch size of `16`.

### Training and evaluation: ordinal regression

Asking ChatGPT it suggested that I should treat this as an ordinal regression
problem instead of a classification problem, using something called [Coral](https://arxiv.org/abs/1901.07884), implemented with a custom subclass of `PreTrainedModel`. I
implemented this in [./ordinal-regression.ipynb](./ordinal-regression.ipynb).

At this point I started measuring QWK and MAE. The best results I got were:

- QWK: `0.56`
- MAE: `1.23`

which are better than the baseline, but not by a lot.

### Next steps

It's likely that I might need to change something fundamental in the approach.
I don't think that tuning the hyperparameters is going to get me much further,
but maybe unfreezing more layers will help.

From this, I learned how to split a problem and the importance of choosing the right
metrics in advance, by reminding myself what kind of problem is being solved.

After this, I would like to inform myself more about the right approach and
potentially get to a MAE of less than `1.0` and a QWK of more than `0.5`.
