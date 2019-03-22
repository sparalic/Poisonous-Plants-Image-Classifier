
onst AppConfig = 

{
    title : "Top 10 Poisonous Plants in the US"
    ,host : "http://poisonousplantsus.herokuapp.com"
};

const description = `
This mobile app was developed by 
- [Sparkle Russell-Puleri](https://www.linkedin.com/in/sparkle-russell-puleri-ph-d-a6b52643/)
- [Dorian Puleri](https://www.linkedin.com/in/dorian-puleri-ph-d-25114511/)
Are you an avid outdoor explorer? Well this classifier can help you identify America's top 10 poisonous plants, so that you can spend your time enjoying the outdoors.
The model was developed using the [images downloaded from google using a custom pipeline] and the [fastai](https://github.com/fastai/fastai) deep learning library, which is built on PyTorch.
To learn more about deep learning, consider taking the [fast.ai course](https://www.fast.ai/) taught by [Jeremy Howard](https://www.fast.ai/about/#jeremy) and [Rachel Thomas](https://www.fast.ai/about/#rachel).
`;
export {AppConfig, description}
