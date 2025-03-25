# candle-chinese-clip

## Running on cpu

```bash
$ cargo run --example chinese_clip --release -- --images "candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg","candle-examples/examples/yolo-v8/assets/bike.jpg" --cpu --sequences "一场自行车比赛","两只猫的照片","一个机器人拿着蜡烛"

> Results for image: candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg
>
> 2025-03-25T19:22:01.325177Z  INFO chinese_clip: Probability: 0.0000% Text: 一场自行车比赛 
> 2025-03-25T19:22:01.325179Z  INFO chinese_clip: Probability: 0.0000% Text: 两只猫的照片 
> 2025-03-25T19:22:01.325181Z  INFO chinese_clip: Probability: 100.0000% Text: 一个机器人拿着蜡烛 
> 2025-03-25T19:22:01.325183Z  INFO chinese_clip: 
> 
> Results for image: candle-examples/examples/yolo-v8/assets/bike.jpg
> 
> 2025-03-25T19:22:01.325184Z  INFO chinese_clip: Probability: 100.0000% Text: 一场自行车比赛 
> 2025-03-25T19:22:01.325186Z  INFO chinese_clip: Probability: 0.0000% Text: 两只猫的照片 
> 2025-03-25T19:22:01.325187Z  INFO chinese_clip: Probability: 0.0000% Text: 一个机器人拿着蜡烛 
```

## Running on metal

```bash 
$ cargo run --features metal --example chinese_clip --release -- --images "candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg","candle-examples/examples/yolo-v8/assets/bike.jpg" --cpu --sequences "一场自行车比赛","两只猫的照片","一个机器人拿着蜡烛"

> Results for image: candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg
>
> 2025-03-25T19:22:01.325177Z  INFO chinese_clip: Probability: 0.0000% Text: 一场自行车比赛 
> 2025-03-25T19:22:01.325179Z  INFO chinese_clip: Probability: 0.0000% Text: 两只猫的照片 
> 2025-03-25T19:22:01.325181Z  INFO chinese_clip: Probability: 100.0000% Text: 一个机器人拿着蜡烛 
> 2025-03-25T19:22:01.325183Z  INFO chinese_clip: 
> 
> Results for image: candle-examples/examples/yolo-v8/assets/bike.jpg
> 
> 2025-03-25T19:22:01.325184Z  INFO chinese_clip: Probability: 100.0000% Text: 一场自行车比赛 
> 2025-03-25T19:22:01.325186Z  INFO chinese_clip: Probability: 0.0000% Text: 两只猫的照片 
> 2025-03-25T19:22:01.325187Z  INFO chinese_clip: Probability: 0.0000% Text: 一个机器人拿着蜡烛 
```
