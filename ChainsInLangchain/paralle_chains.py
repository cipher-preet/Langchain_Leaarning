from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()

model2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

paralled_chain = RunnableParallel(
    {"notes": prompt1 | model1 | parser, 
    "quiz": prompt2 | model2 | parser}
)

merge_chain = prompt3 | model1 | parser

chain = paralled_chain | merge_chain

text = """Generative artificial intelligence (Generative AI, GenAI,[1] or GAI) is a subfield of artificial intelligence that uses generative models to produce text, images, videos, or other forms of data.[2][3][4] These models learn the underlying patterns and structures of their training data and use them to produce new data[5][6] based on the input, which often comes in the form of natural language prompts.[7][8]

Generative AI tools have become more common since an "AI boom" in the 2020s. This boom was made possible by improvements in transformer-based deep neural networks, particularly large language models (LLMs). Major tools include chatbots such as ChatGPT, Copilot, Gemini, Grok, and DeepSeek; text-to-image models such as Stable Diffusion, Midjourney, and DALL-E; and text-to-video models such as Sora.[9][10][11][12] Technology companies developing generative AI include OpenAI, Anthropic, Meta AI, Microsoft, Google, DeepSeek, and Baidu.[7][13][14]

Generative AI has raised many ethical questions. It can be used for cybercrime, or to deceive or manipulate people through fake news or deepfakes.[15] Even if used ethically, it may lead to mass replacement of human jobs.[16] The tools themselves have been criticized as violating intellectual property laws, since they are trained on and emulate copyrighted works of art.[17]

Generative AI is used across many industries. Examples include software development,[18] healthcare,[19] finance,[20] entertainment,[21] customer service,[22] sales and marketing,[23] art, writing,[24] fashion,[25] and product design.[26]

History
Main article: History of artificial intelligence
Early history
The first example of an algorithmically generated media is likely the Markov chain. Markov chains have long been used to model natural languages since their development by Russian mathematician Andrey Markov in the early 20th century. Markov published his first paper on the topic in 1906,[27][28] and analyzed the pattern of vowels and consonants in the novel Eugeny Onegin using Markov chains. Once a Markov chain is learned on a text corpus, it can then be used as a probabilistic text generator.[29][30]

Computers were needed to go beyond Markov chains. By the early 1970s, Harold Cohen was creating and exhibiting generative AI works created by AARON, the computer program Cohen created to generate paintings.[31]

The terms generative AI planning or generative planning were used in the 1980s and 1990s to refer to AI planning systems, especially computer-aided process planning, used to generate sequences of actions to reach a specified goal.[32][33] Generative AI planning systems used symbolic AI methods such as state space search and constraint satisfaction and were a "relatively mature" technology by the early 1990s. They were used to generate crisis action plans for military use,[34] process plans for manufacturing[32] and decision plans such as in prototype autonomous spacecraft.[35]

Generative neural networks (2014-2019)
See also: Machine learning and deep learning

Above: An image classifier, an example of a neural network trained with a discriminative objective. Below: A text-to-image model, an example of a network trained with a generative objective.
Since inception, the field of machine learning has used both discriminative models and generative models to model and predict data. Beginning in the late 2000s, the emergence of deep learning drove progress, and research in image classification, speech recognition, natural language processing and other tasks. Neural networks in this era were typically trained as discriminative models due to the difficulty of generative modeling.[36]

In 2014, advancements such as the variational autoencoder and generative adversarial network produced the first practical deep neural networks capable of learning generative models, as opposed to discriminative ones, for complex data such as images. These deep generative models were the first to output not only class labels for images but also entire images.

In 2017, the Transformer network enabled advancements in generative models compared to older Long-Short Term Memory models,[37] leading to the first generative pre-trained transformer (GPT), known as GPT-1, in 2018.[38] This was followed in 2019 by GPT-2, which demonstrated the ability to generalize unsupervised to many different tasks as a Foundation model.[39]

The new generative models introduced during this period allowed for large neural networks to be trained using unsupervised learning or semi-supervised learning, rather than the supervised learning typical of discriminative models. Unsupervised learning removed the need for humans to manually label data, allowing for larger networks to be trained.[40]

Generative AI boom (2020-)
Main article: AI boom

AI generated images have become much more advanced.
In March 2020, the release of 15.ai, a free web application created by an anonymous MIT researcher that could generate convincing character voices using minimal training data, marked one of the earliest popular use cases of generative AI.[41] The platform is credited as the first mainstream service to popularize AI voice cloning (audio deepfakes) in memes and content creation, influencing subsequent developments in voice AI technology.[42][43]

In 2021, the emergence of DALL-E, a transformer-based pixel generative model, marked an advance in AI-generated imagery.[44] This was followed by the releases of Midjourney and Stable Diffusion in 2022, which further democratized access to high-quality artificial intelligence art creation from natural language prompts.[45] These systems demonstrated unprecedented capabilities in generating photorealistic images, artwork, and designs based on text descriptions, leading to widespread adoption among artists, designers, and the general public.

In late 2022, the public release of ChatGPT revolutionized the accessibility and application of generative AI for general-purpose text-based tasks.[46] The system's ability to engage in natural conversations, generate creative content, assist with coding, and perform various analytical tasks captured global attention and sparked widespread discussion about AI's potential impact on work, education, and creativity.[47]

In March 2023, GPT-4's release represented another jump in generative AI capabilities. A team from Microsoft Research controversially argued that it "could reasonably be viewed as an early (yet still incomplete) version of an artificial general intelligence (AGI) system."[48] However, this assessment was contested by other scholars who maintained that generative AI remained "still far from reaching the benchmark of 'general human intelligence'" as of 2023.[49] Later in 2023, Meta released ImageBind, an AI model combining multiple modalities including text, images, video, thermal data, 3D data, audio, and motion, paving the way for more immersive generative AI applications.[50]

In December 2023, Google unveiled Gemini, a multimodal AI model available in four versions: Ultra, Pro, Flash, and Nano.[51] The company integrated Gemini Pro into its Bard chatbot and announced plans for "Bard Advanced" powered by the larger Gemini Ultra model.[52] In February 2024, Google unified Bard and Duet AI under the Gemini brand, launching a mobile app on Android and integrating the service into the Google app on iOS.[53]

In March 2024, Anthropic released the Claude 3 family of large language models, including Claude 3 Haiku, Sonnet, and Opus.[54] The models demonstrated significant improvements in capabilities across various benchmarks, with Claude 3 Opus notably outperforming leading models from OpenAI and Google.[55] In June 2024, Anthropic released Claude 3.5 Sonnet, which demonstrated improved performance compared to the larger Claude 3 Opus, particularly in areas such as coding, multistep workflows, and image analysis.[56]


Private investment in AI (pink) and generative AI (green).
According to a survey by SAS and Coleman Parkes Research, China has emerged as a global leader in generative AI adoption, with 83% of Chinese respondents using the technology, exceeding both the global average of 54% and the U.S. rate of 65%. This leadership is further evidenced by China's intellectual property developments in the field, with a UN report revealing that Chinese entities filed over 38,000 generative AI patents from 2014 to 2023, substantially surpassing the United States in patent applications.[57]"""

result = chain.invoke({"text": text})

print(result)