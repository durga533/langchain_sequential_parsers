from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Annotated,Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env vars
load_dotenv()

# Initialize model with API key
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

review_input = '''
I was eagerly waiting for this film and I was not disappointed.

Baahubali is one of the most visually impressive films that have come from India. Everything about this film is good - Story, direction, beautiful cinematography, haunting score and amazing visuals (though they lack some details at some places).

I didn't know who Rajamouli was, basically because I am from North India and I don't follow south Indian cinema. But I am sure he is someone you should know about. SS Rajamouli is amazing. He has made something that we can be proud of. I was sure that the VFX would be good however I had a doubt about the story. Rajamouli proved me wrong with this film. The story and the execution is really good. Story takes elements from Indian mythologies (specially from the life of Lord Krishna) . You can not expect more. He took a big risk and made it big. This is what we need, good quality cinema. Salman fans should learn something from it.

All the actors are good and perform their part really well(except Tamanna. She tries too hard but fails miserably). In fact, Tamanna has been used as a plot device (A MacGuffin as Alfred Hitchcock used to call it). Prabhas is okay in the beginning but transforms into a behemoth during the second half. This guy is a natural. He doesn't act exceptionaly but his presence is enough to give you the feels.

The background score needs special mention because it is amazing. It is different from anything you have ever heard in India. The score that plays during war is haunting. (Clearly inspired from the LotR)

I would be lying if I didn't say that Rajamouli spent a lot of time watching The Lord of The Rings. Yes, some scenes will remind you of 300 and the Lord of The Rings but believe me it is completely different. He doesn't copy anything, he uses it as a source of inspiration. There is only one scene that I remember from LotR which has been used in Baahubali. But denying that he was influenced by Peter Jackson would be a big lie. Its a great thing that he was influenced by such a great film otherwise we wouldn't have Baahubali.

The quality of graphics work is very good. In fact its insane for such a cheap price given that Hollywood uses big budget for the same. The battle sequence is SPECTACULAR and reminded me of Battle of Helm's Deep and Pelennor fields from LotR.

But the movie is not without some minor problems. There are unnecessary songs which ruin the experience (I laughed so hard when Prabhas started dancing randomly. Everyone was looking at me.) and some parts are dragged (specially the first half) . Also there are continuity problems and some actors are over-the-top. Cut down the songs and edit the movie and voilà we will have a SPECTACULAR film.

Baahubali is not the answer to 300, Gladiator or The Lord of The Rings because saying so would be foolish. Don't compare this movie to the Hollywood ones because we still have a long way to go. It's not the Indian 300, its Indian "Baahubali".

So in the end, I would like to say - Go and watch it instead of watching the crap that Bollywood has been producing lately. Its something to be proud of. Book your tickets and watch this Indian grand spectacle. You wont regret it. P.S: I want to see Rajamouli doing "Mahabharata" next.

'''


prompt1 = PromptTemplate(
    
template = 'This is a movie review in Englis. Translate this review into Telugu. Review :{review}',
input_variables= ['review']
)


prompt2 = PromptTemplate(
    template = 'Summarize the Telugu review in a short paragraph in Telugu :{review}',
    input_variables= ['review']
)

parser = StrOutputParser()

main_chain = prompt1 | model | parser | prompt2 | model | parser

result = main_chain.invoke({'review': review_input})

print(result)

print(main_chain.get_graph().draw_ascii())


