# README


## Question

Can we predict a player’s performance/worth in an upcoming season based
on their previous performance stats and other metrics?

Baseball has always been a game of numbers. However, with the rise of
technology, the methods of how we obtain those numbersr has been
evolving fast. Analyzing data and predicting a player’s performances in
the future is a crucial step for a team in their roster management. In
this project, I will attempt to predict a given player’s worth in an
upcoming season given their past performances and age, among other
factors.

## Data Dimensions

*Statcast* 45 columns x 2492 rows (players with 300+ plate appearances
in any given season from 2015-2024 except 2020) + 142 rows (qualified
hitters from shortened 2020 season)

*Baseball Reference* 34 columns x 1100 - 1300 rows x 10 years
(2015-2024)

We will merge the Baseball Reference dataset onto the Statcast dataset,
deleting any entries that don’t appear in Statcast. We will only keep
the WAR, Runs, OPS+, rOBA, and Rbat+ columns from Baseball Reference.

## Data Dictionary

**Rk.** - Index; irrelevant

**Player** - Name of player

**Year** - Season year

**Age** - Age of player

**AB** - At-Bats

**PA** - Plate Appearances

**H** - Hits

**2B** - Doubles

**3B** - Triples

**HR** - Home Runs

**SO** - Strikeouts

**BB** - Base on Balls/Walks

**K%** - Strikeout percentage

**BB%** - Base on Balls percentage

**AVG** – Batting average (Hits / At-Bat)

**SLG** - Slugging (1B + 2\*2B + 3\*3B + 4\*HR/At Bats)

**OBP** - On-Base Percentage (H + BB + HBP)/(At Bats + BB + HBP + SF)

**OPS** - On-Base + Slugging Percentage

**ISO** - Isolated Power (ability to hit for extra-base hits)

**BABIP** - Batting average on balls in play

**RBI** - Runs Batted In

**G** - Games played

**HBP** - Hit By Pitch

**xBA** - Expected batting average

**xSLG** - Expected slugging percentage

**wOBA** - Weighted on-base average

**xwOBA** - Expected weighted on-base average

**xOBP** - Expected on-base percentage

**xISO** - Expected isolated power

**wOBACON** - Weighted on-base average on contact

**xwOBACON** - Expected weighted on-base average on contact

**BACON** - Batting average on contact

**xBACON** - Expected batting average on contact

**BA - xBA** - Real batting average - expected batting average

**SLG - xSLG** - Real slugging percentage - expected slugging percentage

**wOBA - xwOBA** - Real weighted on-base average - expected weighted
on-base percentage

**Fast Swing %** - 75 mph or higher bat speed

**LA Sweet-Spot %** - Launch angle between 8 and 32 degrees

**Barrel%** - The batted ball has an expected .500 batting average and
1.500 slugging percentage

**Hard Hit %** - Exit velocity of 95 mph or higher

**EV50** - Average of the hardest 50% of batted balls

**Adjusted EV** - Averages maximum(88, actual exit velocity)

**Whiff %** - Number of pitches swung at and missed / total number of
swings

**Swing %** - Percentage of pitches a batter swings at

**WAR** - Wins Above Replacement

**R** - Runs

**OPS+** - 100\*\[OBP/league OBP + SLG/league SLG - 1\], OPS adjusted to
player’s ballpark

**rOBA** - A measure of a player’s offensive contributions, weighted in
proportion to each event’s actual run value

**Rbat+** - Batting runs as computed for WAR, but indexed to the
environment the player played in, where 100 is league average

## Source

Statcast -
[https://baseballsavant.mlb.com/https://baseballsavant.mlb.com/](https://baseballsavant.mlb.com/)

Baseball Reference - <https://stathead.com/baseball/>

Data collected from all MLB games in 2015 - 2024 (Statcast era)

## Glimpse

``` r
library(tidyverse)
```

    ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ✔ ggplot2   3.5.2     ✔ tibble    3.2.1
    ✔ lubridate 1.9.4     ✔ tidyr     1.3.1
    ✔ purrr     1.0.4     
    ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ✖ dplyr::filter() masks stats::filter()
    ✖ dplyr::lag()    masks stats::lag()
    ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
data <- read_csv("/Users/suisakai/STAT155/Project 1/Data/Statcast.csv")
```

    Rows: 2492 Columns: 45
    ── Column specification ────────────────────────────────────────────────────────
    Delimiter: ","
    chr  (1): last_name, first_name
    dbl (44): player_id, year, player_age, ab, pa, hit, single, double, triple, ...

    ℹ Use `spec()` to retrieve the full column specification for this data.
    ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
glimpse(data)
```

    Rows: 2,492
    Columns: 45
    $ `last_name, first_name` <chr> "Hunter, Torii", "Ortiz, David", "Rodriguez, A…
    $ player_id               <dbl> 116338, 120074, 121347, 133380, 134181, 136860…
    $ year                    <dbl> 2015, 2015, 2015, 2015, 2015, 2015, 2015, 2015…
    $ player_age              <dbl> 39, 39, 39, 37, 36, 38, 36, 36, 38, 36, 36, 41…
    $ ab                      <dbl> 521, 528, 523, 475, 567, 478, 331, 379, 407, 5…
    $ pa                      <dbl> 567, 614, 620, 516, 619, 531, 378, 408, 436, 5…
    $ hit                     <dbl> 125, 144, 131, 117, 163, 132, 73, 98, 122, 116…
    $ single                  <dbl> 81, 70, 75, 68, 109, 78, 44, 69, 88, 76, 60, 7…
    $ double                  <dbl> 22, 37, 22, 31, 32, 34, 16, 18, 24, 24, 17, 5,…
    $ triple                  <dbl> 0, 0, 1, 1, 4, 1, 1, 1, 1, 3, 0, 6, 0, 2, 0, 5…
    $ home_run                <dbl> 22, 37, 33, 17, 18, 19, 12, 10, 9, 13, 14, 1, …
    $ strikeout               <dbl> 105, 95, 145, 68, 65, 85, 84, 88, 37, 86, 80, …
    $ walk                    <dbl> 35, 77, 84, 31, 41, 45, 38, 24, 19, 44, 34, 31…
    $ k_percent               <dbl> 18.5, 15.5, 23.4, 13.2, 10.5, 16.0, 22.2, 21.6…
    $ bb_percent              <dbl> 6.2, 12.5, 13.5, 6.0, 6.6, 8.5, 10.1, 5.9, 4.4…
    $ batting_avg             <dbl> 0.240, 0.273, 0.250, 0.246, 0.287, 0.276, 0.22…
    $ slg_percent             <dbl> 0.409, 0.553, 0.486, 0.423, 0.453, 0.471, 0.38…
    $ on_base_percent         <dbl> 0.293, 0.360, 0.356, 0.297, 0.334, 0.337, 0.30…
    $ on_base_plus_slg        <dbl> 0.702, 0.913, 0.842, 0.720, 0.787, 0.808, 0.68…
    $ isolated_power          <dbl> 0.169, 0.280, 0.236, 0.177, 0.166, 0.195, 0.16…
    $ babip                   <dbl> 0.258, 0.264, 0.278, 0.253, 0.295, 0.297, 0.25…
    $ b_rbi                   <dbl> 81, 108, 86, 75, 83, 67, 42, 41, 49, 41, 43, 2…
    $ b_game                  <dbl> 139, 146, 151, 137, 143, 133, 88, 117, 113, 14…
    $ b_hit_by_pitch          <dbl> 6, 0, 6, 5, 3, 2, 3, 4, 7, 0, 2, 0, 7, 10, 6, …
    $ xba                     <dbl> 0.229, 0.301, 0.247, 0.240, 0.295, 0.274, 0.24…
    $ xslg                    <dbl> 0.370, 0.616, 0.494, 0.405, 0.482, 0.448, 0.42…
    $ woba                    <dbl> 0.304, 0.379, 0.361, 0.309, 0.337, 0.346, 0.30…
    $ xwoba                   <dbl> 0.290, 0.420, 0.368, 0.304, 0.360, 0.346, 0.33…
    $ xobp                    <dbl> 0.285, 0.388, 0.355, 0.293, 0.346, 0.340, 0.32…
    $ xiso                    <dbl> 0.142, 0.314, 0.247, 0.164, 0.186, 0.174, 0.18…
    $ wobacon                 <dbl> 0.343, 0.418, 0.425, 0.329, 0.352, 0.382, 0.33…
    $ xwobacon                <dbl> 0.325, 0.474, 0.436, 0.323, 0.380, 0.381, 0.38…
    $ bacon                   <dbl> 0.297, 0.326, 0.340, 0.284, 0.320, 0.331, 0.28…
    $ xbacon                  <dbl> 0.286, 0.364, 0.338, 0.279, 0.333, 0.334, 0.32…
    $ xbadiff                 <dbl> 0.011, -0.028, 0.003, 0.006, -0.008, 0.002, -0…
    $ xslgdiff                <dbl> 0.039, -0.063, -0.008, 0.018, -0.029, 0.023, -…
    $ wobadiff                <dbl> 0.014, -0.041, -0.007, 0.005, -0.023, 0.000, -…
    $ fast_swing_rate         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
    $ sweet_spot_percent      <dbl> 28.5, 34.8, 31.4, 33.5, 35.7, 34.1, 35.2, 31.2…
    $ barrel_batted_rate      <dbl> 5.0, 13.1, 10.9, 5.6, 5.5, 5.8, 7.9, 4.5, 3.5,…
    $ hard_hit_percent        <dbl> 34.9, 49.1, 43.9, 34.5, 40.4, 41.9, 40.7, 30.8…
    $ avg_best_speed          <dbl> 98.56340, 102.85113, 101.38114, 97.85126, 99.2…
    $ avg_hyper_speed         <dbl> 93.39348, 96.05306, 95.01438, 92.94476, 93.842…
    $ whiff_percent           <dbl> 23.1, 23.2, 32.0, 17.9, 16.8, 18.1, 24.2, 24.4…
    $ swing_percent           <dbl> 53.4, 44.7, 43.9, 52.9, 48.1, 45.4, 37.9, 51.4…
