title: "A New Beginning - Options"
date: 2019-09-04
tags: [Options, Derivatives, Options Bounds]
excerpt: "A new beginning for my blog. I am no longer going to post about DS related subjects, I get enough of that at my job. I will now use this as a place to discuss my new hoppy; Networks in Finance, Economics and Game Theory. While this post doesn't involve networks, many will. This is just a starting point."
mathjax: "true"

# Intro

So I haven't updated this blog in quite some time (almost two year!). Reasons for this include I switched jobs at that point and in my current I am actually working on real data science projects so I haven't had as much desire to write blog posts on DS methods. I have also had a lot of work to do around our house and yard which has taken a lot of my time. Recently though, I have become interested in finance and economics; in particular, the use of network theory in those fields. So now as a hobby, I will use this blog to study finance, economics and game-theory. While many posts will not involve network theory, my goal is to understand how network theory is used in these areas, such as financial market risk and contagions, social network effects on loan uptake, and many other topics. But first, I will start with some basics. One area of interest to start with are assets type and how to price them. This post will start this topic by looking at options and their price bounds. I think this is an intersting place to start as it provides insight on an important asset type and how they work as well as an introdution to arbitrage. I will discuss many more topics and posts may seem scatter shot in that there won't be much continuity between posts. But again, this is simply a fun hobby of mine.


# Options  - Price Bounds

An option is a type of financial asset that derives it's value from some underlying security (stock, commodity, etc.). Hence, an option is part of a group of assets called 'derivatives'(also includes futures). An option gives the owner the option to purchase (or sell) some amount of the underlying security at a given price known as the *strike price* at a time in the future. An optioin to purchase the underlying security is known as a *call option* while one to sell the security is called a *put option*.

The time at which the option can be exercised depends on the type of option. For instance a *European option* can only be exercised at it's experation date while and *American option* can be exercised at any point up to it's experation date. As the name of the asset imples, options do not have to be exercised.

As an example, consider a European option to buy some stock at \\$150 per share in three months. Suppose the stock in three months rises to $170, then you would profit \\$20 per share. However, if the stock declined in value to say \\$140, given that this is an option, you would not exercise the option and only lose the price paid for the option. The question then is what price should be paid for the option?


Some definitions that will be helpful

**long position** - buy a security or derivative with the expectation that it will rise in value.

**short postion** - buy a security or derivative with the expectation that it will decrease in value.

## Bounds on Options Prices

To start, we can look for some price bouonds on the options. We will be looking primarily at European and American options for this discussion. Note, the we distinguish between American and European option prices using capital letters for American and lower case for European. Also, we use subscripts to determine whether the option price refers to a call or put option. So \\(O_c\\) is the price of an american call option while \\(o_c\\) is a European call option.

### Upper Bounds
#### Calls

Call options give the owner of the option the right to buy a given stock at the strike price, \\(K\\), at some point in the future. In this case, the upper bound on the option price, is the current stock price \\(S_0\\). If \\(c\\) is the price of the call option, then

$$ o_c \leq S_0 \text{ and }O_c \leq S_0 $$

This is true for either European or American options. To see why, suppose \\(o_c > S_0\\). In this case, you would be able to purchase the stock now and sell call options on the stock (short position in the call option). The payoff would be \\(o_c-S_0 >0\\). Since you already own the stock you may need to sell the purchaser of the call, there is no future possibility of loss. This type of riskless profit is known as *arbitrage*. These situations supposed happen, but are short lived as prices will adjust quickly. As the temporal aspect of the call doesn't matter in this situation, this upper bound is true for both European and American options.

#### Puts

Put options give the owner the right to sell a stock at a given strike price, \\(K\\), at some point in the future. As such, the upper bound for an American put option is the strike price,

$$ O_p \leq K $$

For a European put option, the upper bound is slightly different. As the time period for the European put option is fixed, we must use the present value of the strike price. Suppose a constant, nominal interest rate \\(r\\) compounded continuously over time period \\(T\\), then the upper bound is

$$ o_p \leq Ke^{-rT} $$

This is not neccessary for the American put since it can be exercised immediately \\((T=0)\\), hence the present value would just be \\(K\\).
To see why this is the upper bound, consider a European option such that \\(o_p > Ke^{-rT}\\). We could then write a put option contract with strike price \\(K\\) and invest the proceeds into an asset that earns the riskless interest rate of \\(r\\). Then at time \\(T\\), we have \\(o_pe^{rT}>K\\) and a riskless profit of \\(o_pe^{rT}-K>0\\), which again is arbitrage.

### Lower Bounds

For lower bounds, we first make the assumption that the stock does not pay a dividend, though we will consider dividends later.

#### Calls

For a European call option, consider the difference between the current stock price and the present value of the strike price

$$ S_0 - Ke^{-rT} = x $$

Now suppose \\( o_c < x \\). In this case, we can short the stock (that is, borrow one and sell it) and buy the option so that we have cash

$$ S_0 - o_c = y > Ke^{-rT}\text{ and } o_c < x $$

We can then invest the cash \\(y\\) earning the riskfree rate \\(r\\). At time time \\(T\\) we have cash \\(ye^{rT}\\) and we can purchase the stock at the strike price \\(K\\).  Since \\(y>Ke^{-rT}\\), then \\(ye^{rT} > K\\) so that the resulting risk free profit of this strategy is

$$ ye^{rT} - K > 0 $$

We therefore have arbitrage, so the price \\(S_0 - Ke^{-rT}\\) is a lower bound on \\(o_c\\). For American options, since they can be exercised at any point up to the expiration date, we again don't discount the strike price to find a lower bound, thus

$$ S_0 - K \leq O_c $$

But this is not quite the whole story, if \\(K > S_0 \\), one would simply buy the stock now. Therefore the lower bounds are actually

$$ o_c \geq \max(S_0-Ke^{-rT},0) \text{ and } O_c \geq \max(S_0 - K, 0) $$

#### Puts

For European put options, consider now the discounted strike price less the current stock price

$$ Ke^{-rT} - S_0 = x $$

Now suppose that \\(o_p < x\\). In this case, we could borrow the amount $S_0 + o_p = y$, with interest \\(r\\), and purchase both the stock and the option. Since \\( y < KE^{-rT}\\), then at time \\(T\\), we have

$$ ye^{rT} < K \implies 0 < K - ye^{rT} $$

So we see that if the stock price is less than \\(K\\), we can exercise the options and gain profit \\(K-ye^{rT}\\). If the the stock is higher than \\(K\\), the option is discarded, the stock is sold and our profit is even greater. So once again, we have arbitrage.

Similar call options, the actual lower bounds are

$$ o_p \geq \max(Ke^{-rT}-S_0,0) \text{  and  } O_p \geq \max(K-S_0,0) $$

That is, if the put price is less than the strike price, it is better to go ahead and buy the stock.

### Example

For a concrete real world example, let's look at the current stock price of a stock, for instance Microsoft (this is from options chain info on Nasdaq). I was hoping to get some historical options data and stock prices to look at ho the price changes over time, but historical data is expensive. I am working on a webscraper to help gather data on my own for this blog.

Currently, the last close price for MSFT on July 23, 2021 was \\$289.67. A put option with excercise date July 30 and strike price \\$292.50 the bounds are

$$ 292.50 - 289.67 = 2.83 \leq O_p \leq 289.67 $$

The last trading price for this option was \\$3.00, close to the lower bound. What about for a strike price of \\$270? Since this price is below the current price, we would expect the option price to be \\$0, or not being sold, which is exactly what is seen in th options chain.

## End Note

This concludes my basic intro to options and how to determine the bounds on their prices. This discussion was based partially on Chapter 11 in *Options, Futures, and Other Derivatves* by John C Hull. In a future post I am working on, I will discuss pricing options in a risk-neutral manner using the **Binomial Asset Pricing Model**. This is an intereresting binomial tree based approach in which options can only take on one of two prices in the future (hence binomial). I will look at both the single period and multiperiod versions. Additonal posts are in the works, currently one on game-theory, one on value-at-risk and another on using netowkrs to study risk in financial markets. Hopefully I will update more often!




```python

```
