# AI Builders Week 5

Download dataset:

```bash
uv run kaggle datasets download -d riyosha/letterboxd-movie-reviews-90000 -p data
unzip ./data/letterboxd-movie-reviews-90000.zip -d ./data
```

## Objective

To fine-tune a model to receive a letterbox review and predict the star rating.

## Data preparation

In [./explore-data.ipynb] I prepped the data:

- filtered out non-english reviews
- checked the distribution of ratings
- sampled 200 reviews per rating (there are less than 300 ratings with half a star, so in order to keep things distributed I set a max of 200).
- split into training, validation and test sets.

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
(with different hyperparameters and models and other things) I didn't get an
accuracy of more than `0.35`, being around 3 times better than random guessing
(which would be `0.1` for 10 different ratings)

