# Where be the aliens? Game theory, Dark Forest, and Python

Contents:
Intro 

1. But where is everybody?
   1. Fermi Paradox and Drake equation
   2. Different solutions
      1. The Great Filter, Rare Earth, Zoo Hypothesis, Simulation Hypothesis, etc.
2. Dark Forest 
   1. Liu Cixin's trilogy
   2. The Dark Forest theory 
   3. Assumptions
      1. Comunication, Elimitation, Speed
3. Space prisoners  
   1. Prisoner's Dilemma
   2. The Dark Forest game
      1. Nash equilibrium
      2. Fragile deterrence
   3. Iterated version, is it possible?
4. Extending to N civilizations
   2. Earth's chances
      1. How loud have we been?
      2. How far can we hear?
5. So what to do? 
   1. Stay quiet
   2. Expand 
6. Criticism of Dark Forest theory

# Introduction 

Hi There! I love to think about the universe and the mysteries it holds. One of the most intriguing questions is the Fermi Paradox, which asks why we haven't seen any evidence of extraterrestrial civilizations. In this article, I will share my attempts to apply game theory to one of the solutions of the Fermi Paradox, Dark Forest, which derives its name from [Liu Cixin's 2008 novel](https://www.goodreads.com/book/show/23168817-the-dark-forest). I will explain the theory, its assumptions, and how it relates to game theory. Let's dive in for a daily dose of existential dread and cosmic horror!
<!-- 
I will also show you how to simulate the Dark Forest game in Python. -->

# But where is everybody?

There are over [100 billion stars](https://www.space.com/25959-how-many-stars-are-in-the-milky-way.html) in our galaxy, and over [2 trillion galaxies](https://www.skyatnightmagazine.com/space-science/how-many-galaxies-in-universe) in the observable universe. How come we haven't seen any evidence of life outside Earth? The famous [Fermi Paradox](https://en.wikipedia.org/wiki/Fermi_paradox) from 1950 asks this question. The apparent contradiction between the high probability of extraterrestrial civilizations' existence and the lack of evidence for, or contact with, such civilizations is the paradox. The [Drake equation](https://en.wikipedia.org/wiki/Drake_equation) is a probabilistic argument used to estimate the number of active, communicative extraterrestrial civilizations in the Milky Way galaxy. The equation was formulated in 1961 by Frank Drake, not for purposes of quantifying the number of civilizations, but as a way to stimulate scientific dialogue at the first scientific meeting on the search for extraterrestrial intelligence (SETI). The Drake equation is:

$$N = R^* \cdot f_p \cdot n_e \cdot f_l \cdot f_i \cdot f_c \cdot L$$

where:
- $N$ is the number of civilizations in our galaxy with which we might expect to be able to communicate at any given time,
- $R^*$ is the average rate of star formation in our galaxy,
- $f_p$ is the fraction of those stars that have planets,
- $n_e$ is the average number of planets that can potentially support life per star that has planets,
- $f_l$ is the fraction of planets that could support life that actually develop life at some point,
- $f_i$ is the fraction of planets with life that develop intelligent life (civilizations),
- $f_c$ is the fraction of civilizations that develop a technology that releases detectable signs of their existence into space,
- $L$ is the length of time for which such civilizations release detectable signals into space.

Naturally, the equation is highly speculative and depending on the values of the unknown (and unknowable) parameters, the result can be bend in any direction depending on the author's bias. 


## Different solutions

There are many proposed solutions to the Fermi Paradox, some of the most popular are: [The Great Filter](https://en.wikipedia.org/wiki/Great_Filter), [Rare Earth](https://en.wikipedia.org/wiki/Rare_Earth_hypothesis), [Zoo Hypothesis](https://en.wikipedia.org/wiki/Zoo_hypothesis), [Simulation Hypothesis](https://en.wikipedia.org/wiki/Simulation_hypothesis), etc. Without going into details, the Great Filter suggests that there is some unknown step in the evolution of life that is extremely unlikely, and that is why we haven't seen any other civilizations. The Rare Earth hypothesis insists that Earth is a very special place, and that the conditions for life are very rare. The Zoo Hypothesis speculates that there are many civilizations out there, but they are intentionally avoiding communication with Earth to allow for natural evolution. The Simulation Hypothesis suggests that we are living in a simulation, and that is why we haven't seen any other civilizations. However, my favorite solution is the Dark Forest theory, named after the second book in [Liu Cixin's trilogy](https://en.wikipedia.org/wiki/Remembrance_of_Earth%27s_Past).

# Dark Forest

(important assumption -- both civilizations have the ability to destroy each other)

# Space prisoners  

![Image by author](images/c1_c2_prisoners_game.png)

Let's start with an illustration of the decisions that two cosmic civilizations can make, trying to closely follow the Liu Cixin's terminology from the original book. First, a few key definitions:

- $C_n$ - civilization playing the game
- $d_n$ - payoff that civilizations $C_n$ gets for destroying the other civilization
- $f_n$ - payoff that civilizations $C_n$ gets for befriending the other civilization

Civilization $C_1$ starts the game. It has two options to start with: **Broadcast** and **Do nothing**. If it chooses to **Broadcast**, it will send a signal to the universe, revealing its location. If it chooses to **Do nothing**, it will remain silent and hidden. 

If civilization $C_2$ receives a signal from $C_1$ (left branch), is can choose to **Destroy** the sender, ignore it and **Do nothing** or **Broadcast** itself. Destroying the sender would result in an infinitely negative payoff for $C_1$ and a a non-negative payoff $d_2$ for $C_2$. Doing nothing would result in a payoff of $0$ for both civilizations. Finally, broadcasting would reveal the location of $C_2$ for $C_1$ which will act next.

Having recieved a signal from $C_2$, $C_1$ has the same **Destroy** and **Do nothing** options, but also has the option to **Befriend** the other civilization. Destroying $C_2$ naturally would lead to an infinitely negative payoff for $C_2$ and a non-negative payoff $d_1$ for $C_1$. Doing nothing would once again result in a payoff of $0$ for both civilizations, and befriending would result in a non-negative payoff of $f_1$ for $C_1$ and $f_2$ for $C_2$.

The right branch of the tree represents the case where $C_1$ decides to stay silent, so the game essentially starts over with $C_2$.

Now let's try to find Nash equilibria for this game. We'll do this analytically, and then we'll simulate the game in Python. 

## Analytical solution

Let's start untangling the game tree by taking a closer look at the left branch where **broadcast from $C_1$ has already happened**. We will start from the bottom and work our way up.

After $C_2$ broadcasts its location, it puts itself in a very vulnerable position. Now the fate of $C_2$ is in the hands of $C_1$. From the payoffs we can see that depending on the values of $d_1$ and $f_1$, $C_1$ could choose any of the three options. If $d_1 > f_1 \geq 0$, it would be rational for $C_1$ to destroy $C_2$. If $0 \leq d_1 < f_1$, it would be rational for $C_1$ to befriend $C_2$. Finally, if $d_1 = f_1 = 0$, $C_1$ would be indifferent between destroying, befriending, and doing nothing. Bummer.

To add more certainty, let's get back to Liu Cixin and use his suggestions from the book. Destroying a civilization does not yield any benifits like harvesting the remaining resources. It can only eliminate a potential threat. Moreover, for both Earth and Trisolaris destroying the other civilization would be a pyrric victory. For Trisolaris that would mean losing the last hope for a stable home, and for Earth that would first have a lot of moral complications, and second, would reveal Earth's location to the rest of the universe rendering it even more vulnerable. 

On the other hand, befriending, which originated as deterrance, is fairly beneficial for both civilizations at least in the short run. Earth was gaining new technologies, and Trisolaris was learning more about Earth's culture and history. 

In this settings, we can say that $f_1 > 0 > d_1$ and $f_2 > 0> d_2$. So in the case of $C_2$ broadcasting, $C_1$ would be rational to befriend $C_2$.

Now that we have defined the payoffs, let's formulate a matrix and simplify the game assuming that the civilizations are playing (almost) simultaneously and both have the options to **Destroy**, **Do nothing**, and **Befriend**.


|         | **Destroy**  | **Befriend** | **Do nothing**  |
|---------|----------|------------|-----------|
| **Destroy** | $-\infty, -\infty$ | $d_1, -\infty$    | $d_1, -\infty$ |
| **Befriend** | $-\infty, d_2$    | $f_1, f_2$       | $0, 0$    |
| **Do nothing** | $-\infty, d_2$ | $0, 0$    | $0, 0$ |

In this matrix, the rows represent the actions of civilization $C_1$ and the columns represent the actions of civilization $C_2$. 

Now let's try eliminating dominated strategies. A strategy is dominated if there is another strategy that is always better, regardless of the opponent's choice. Taking a closer look at the *Do nothing* row, we can see that regardless of the choice of $C_2$, $C_1$ would always be better off by choosing *Befriend*. The same is true for the *Do nothing* column so we can eliminate this strategy. from the game. 

The simplified matrix is now:

|         | **Destroy**  | **Befriend** |
|---------|----------|------------|
| **Destroy** | $-\infty, -\infty$ | $d_1, -\infty$    |
| **Befriend** | $-\infty, d_2$    | $f_1, f_2$       |


Continuing the elimination process we can now see that regardless of the choice of $C_2$, $C_1$ would always be better off by choosing *Befriend* and vice versa. So the pure strategy Nash equilibrium of this game is for both civilizations to befriend each other:

|         | **Befriend** |
|---------|------------|
| **Befriend** | $f_1, f_2$|

So should we all just be friends and send greetings left and right? Well, not so fast.

## Weak and Strong civilizations 

<p align="center">
<img src="images/neat_part.png" alt="drawing" style="width:300px;"/>
</p>

Let's introduce the concept of a weak and a strong civilization. A weak civilization is the one that does not have the ability to destroy others. It might not have reached the necessary technological level to do so or its resources are insufficient. A strong civilization, on the other hand, can wipe out others.

Getting back to our game tree, we will assume that $C_1$ is now a weak player and can no longer annihilate anyone. $C_2$ is still strong and can destroy $C_1$ if it chooses to. Moreover, the freidnship payoffs are shifting in favor of $C_2$ as it can now exploit $C_1$ without any fear of retaliation. 

We will assume that if $C_2$ chooses to **Befriend** it would effectively occupy $C_1$ and get all of its resources. So the payoff for $C_2$ would be $F_1 \gg f_2 > 0$, while the payoff for $C_1$ would be similar to complete destruction, let's call it $-D_1$, where $D_1 \rightarrow \infty$.

To make things worse for $C_1$, stronger $C_2$ can now forcefully **Befriend** (occupy) $C_1$ as soon as it learns about its location. 

![Image by author](images/c1_c2_weak_c1.png)


Once again, let's build a payoffs matrix assuming that the initial broadcast of $C_1$ has already happened. This time we can remove the **Destroy** option for $C_1$ as it's no longer available:

|         | **Destroy**  | **Befriend** | **Do nothing**  |
|---------|----------|------------|-----------|
| **Befriend** | $-\infty, d_2$    | $-D_1, F_2$       | $0, 0$    |
| **Do nothing** | $-\infty, d_2$ | $-D_1, F_2$    | $0, 0$ |

As we can see, the outcomes for $C_1$ are the same regardless of the choice of $C_2$. We can simplify the matrix by eliminating the dominated strategies of $C_2$:

|         | **Befriend** |
|---------|------------|
| **Befriend** | $-D_1, F_2$|
| **Do nothing** | $-D_1, F_2$|

So the Nash equilibrium of this game is occupation of $C_1$ by $C_2$:

|         | **Occupy** |
|---------|------------|
| **Become occupied** | $-D_1, F_2$|

From this result alone it's already clear that for a weak civilization, it's better to not initiate contact with a strong one at all. And how could we as a civilization know if a weak or strong counterpart is listening on the other end? Well, that's the neat part. We don't. 


## Incomplete information and why beliefs matter

The result above, devastating for $C_1$, is only true if $C_2$ knows or, rather, *believes*, that $C_1$ is weak. How would the game unfold if despite $C_1$ being weak, $C_2$ would not be sure about it? 

We are now entering a domain of games with incomplete information, where players may not know strategies, payoffs, or "types" of other players. In our case, a civilization may only know its own strength, but not the strength of the other civilization and has to make decisions based on beliefs.

Let's analyse this scenario from a positoin of a civilization receiving the signal, $C_2$. As the civilisation listens to a message from dark corners of the universe, it starts wondering. Who is the sender? What are their intentions? What kind of technological level they had achieved by the time the message was sent? And finally, what should the people of $C_2$ do about it?

As we have seen form the previous outcomes, [peace is rarely an option](https://i.kym-cdn.com/entries/icons/original/000/018/215/cover8.jpg). If $C_2$ decides to reply and reveal itself, its best-case scenario is for $C_1$ to be either equally strong to engage in fragile deterrence or weaker to be occupied. But what if it's stronger? Or if it will have become stronger when the reply reaches them? Is this all a [trap?](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHBvb3dzYmM1OWRvanllcWlsa2N3cWg1bmc3andhcDFoZXkzcml5YSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3ornka9rAaKRA2Rkac/giphy.gif) A way to lure other civilizations out of hiding and then consume them? The uncertainty is terrifying.

Let $p$ be the belief probability that $C_1$ is weaker, $q$ be the belief that it's equally strong and $1-p-q$ that it's stronger. Based on the results form out previous analyses, the payoffs for $C_2$ would be: 
- $F_2$ in case $C_1$ is weaker, ($C_2$ occupies/consumes/destroys $C_1$)
- $f_2$ in case $C_1$ is equally strong (fragile deterrence)
- $-D_2$ in case $C_1$ is stronger ($C_2$ gets occupied/consumed/destroyed)

Putting it all together and using belief probabilities as weights, we get the final payoff for $C_2$ as a function of $p$ and $q$:

$$p \cdot F_2 + q \cdot f_2 + (1-p-q)\cdot(-D_2)$$

Remembering that $D_2 \rightarrow \infty$, it becomes clear that no matter the beliefs of $C_2$, the percpective of being occupied (with a potential of being completely wiped out) would outweigh any potential benefits of replying. So what is the reasonable thing for $C_2$ to do? Obviously nothing, just stay silent and hidden.

But then comes another realization. What if $C_1$ is planning to expand and will eventually detect and threaten $C_2$ anyway? How can $C_2$ prevent this from happening? The answer is simple. Destroy a potential competitor $C_1$ before it has a chance to grow. And if $C_2$ does not yet have the power to do so, it's in their civilization's best interest to start developing the ultimate weapon as soon as possible hoping to stay undetected in the meantime. 

Now all of this is very gloomy and pessimistic. Surely there are friendly aliens out there who are as curious, open-minded, and peaceful as human civilization will hopefully be sometime in the future? Shouldn't we actively try to reach out to them and establish contact? Well, let's see how this would play out from the perspective of the original messenger, $C_1$.


<!-- ## Simulating the game in Python

Now before we finalise our journey through the tree, let's see how we can get to the same result by simulating the game in Python with the help of the `nashpy` library.  -->



# Extending to N civilizations

So far we have been considering a game between two civilizations. But in the observable universe there are billions of stars, planets, and potentially civilizations. We finally extend the game and welcome many more alien participants. $C_1$ still has an option to send our a signal, but now there are countless civilizations listening. If they recieve a signal, they have to make a decision based on the beliefs about the strength of the sender and the potential threats it poses.

Let's introduce a few variables:

- let $p$ be the probaility that that a civilization $C_n$ receiving the message is equal to or weaker than $C_1$
- $(1-p)$ would be the probability that $C_n$ is stronger
- $N$ is the total number of civilizations that got the message from $C_1$
​

Now with those variables in mind, we can say that there should be $pN$ civilizations that are equal to or weaker than $C_1$ and $(1-p)N$ civilizations that are stronger. From our previous analysis we know that the equilibrium outcome for $C_1$ when it encounters a weaker civilization is $F_1$, for an equally strong civilization is $f_1$, and for a stronger civilization is either $-D_1$ if the stronger one decides to occupy, where $D_1 \rightarrow \infty$, or simply $-\infty$ if it immediately decides to destroy $C_1$. 

To make the analysis simpler, let's say that encountering a weaker or equally strong civilization is generally a positive case for $C_1$, and the payoff could be some probability-weighted function of $F_1$ and $f_1$: $P_1(f_1, F_1) \geq 0$. Since there are $pN$ such civilizations, and chances of encountering them are completely independent, the total positive payoff for $C_1$ would be $pN \cdot P_1(f_1, F_1)$.

On the other hand, getting exposed to a stronger civilization ends up in either desctruction or occupation, and the fact that out there might be $(1-p)N$ of strong ones does not make things easier for feeble $C_1$. The negative payoff could be another probability-weighted function, depending on the choice of the stronger civilization. For simplicity, let's say that it's always $-D_1$, so the total payoff for $C_1$ in this case would be $(1-p)N \cdot (-D_1)$.

Combining the two terms together, we get the total payoff for $C_1$ after sending out a signal:
$$
\begin{align*}
\text{Total Payoff} &=  pN \cdot P_1(f_1, F_1) + (1-p)N \cdot (-D_1) \\
                    &= N \cdot (p \cdot P_1(f_1, F_1) - (1-p) \cdot D_1)
\end{align*}
$$

Given that $D_1 \rightarrow \infty$, the term $(1-p) \cdot D_1$ would dominate as long as there is a non-zero probability of encountering a stronger civilization, i.e. $(1-p) > 0$. The total payoff for $C_1$ then tends towards:

$$
\begin{align*}
\text{Total Payoff} &\propto - (1-p) \cdot D_1\\
                    &\propto -\infty
\end{align*}
$$

This means that as long as there is a singe strong civilization capable of figuring out the location of $C_1$, the total payoff for $C_1$ would be infinitely negative. Only a very naive civilization that either thinks it's the strongest in the universe or is completely ignorant of the potential threats would be willing to initiate contact.

But wait, there is a civilization like that. It's us.

<p align="center">
<img src="images/Ben_kenobi.png" alt="drawing" style="width:400px;"/>
</p>


<!-- Important! The civilization that fears the dark forest is bening, but ones that do not fear would exapd, attack, and destroy others and so are evil and must be eliminated.

In the eyes of external observer, who would dare to broadcast their location? -->


# How screwed are we? 

Humanity has been whispering into the void ever since the first radio waves started travelling through space. Most of the signals that leaked into the universe were not intentionally sent to communicate with extraterrestrial civilizations, but rather a byproduct of our technological progress. However, some attempts were symbolically intentional, like the [Arecibo message](https://en.wikipedia.org/wiki/Arecibo_message) or the [Voyager Golden Record](https://en.wikipedia.org/wiki/Voyager_Golden_Record). 

Naturally, the radio waves have been exceptionally weak and chances of them being detected by an alien civilization are slim. We can do some napkin math to estimate first 

But let's imagine that a wise and technologically advanced civilization would invest a lot of resourced into building the most sensitive equipment to detect even the faintest signals from the universe. After all, their survival depends on it. 

- estimate the distance travelled by first radiowaves
- estimate how many starts/planets/civilizations could be in the area
- etc. 

# So what to do?
# Criticism of Dark Forest theory