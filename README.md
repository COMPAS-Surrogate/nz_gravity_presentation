---
author: Avi Vajpeyi
title: Population Inference with active learning
subtitle: NZ Gravity Workshop, Summer 2024
date: February 16 2024
background-image: imgs/background.png
data-background-size: cover

---



## Outline

- Astronomical populations 
- Learning from populations 
- Active learning + Bayesian Optimisation
- Preliminary results 

---


## Astronomical populations 


### Stars 



### Compact binaries 

## Learning from populations

`Population inference`


### The 'inference' loop

search 


### Population model


![](imgs/model.gif)


### Estimated 

[//]: # (---------------------------------------------)

## Active learning

Training a 'surrogate' model


### Improve 'NN' with new data while training


| Start       | 50 iterations |
|-------------|---------------|
| ![al_start] | ![al_end]     |


[al_start]: https://raw.githubusercontent.com/maurock/snake-ga/master/img/notraining.gif
[al_end]: https://raw.githubusercontent.com/maurock/snake-ga/master/img/snake_new.gif



### Bayesian Optimisation

<a href="https://gifyu.com/image/SCD96"><img src="https://s13.gifyu.com/images/SCD96.gif" alt="model" border="0" /></a>

### COMPAS Surrogate + Bayesian Opt loop

<a href="https://gifyu.com/image/SCD94"><img src="https://s13.gifyu.com/images/SCD94.gif" alt="bo al main" border="0" /></a>












[//]: # (---------------------------------------------)
## Preliminary results


### Surrogate training progress


### Parameter estimates



### Posterior-posterior test



### Next steps 

- Simulation studies with various surrogates
- Simulation studies with uncertain data
- Real LVK dataset

[//]: # (---------------------------------------------)