from llm_engineering_lab.tokenization.benchmark import (
    load_tokenizer,
    benchmark_text,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = load_tokenizer(MODEL_NAME)


TEXTS = {
    "English": (
        "Large language models process text as tokens. "
        "Tokenization affects context usage, inference cost, "
        "memory consumption, and latency."
    ),
    "Hindi": (
        "बड़े भाषा मॉडल पाठ को टोकन के रूप में संसाधित करते हैं। "
        "टोकनाइजेशन संदर्भ उपयोग, अनुमान लागत, मेमोरी खपत और विलंबता को प्रभावित करता है।"
    ),
    "Telugu": (
        "పెద్ద భాషా నమూనాలు పాఠ్యాన్ని టోకెన్లుగా ప్రాసెస్ చేస్తాయి. "
        "టోకెనైజేషన్ సందర్భ వినియోగం, అనుమితి ఖర్చు, మెమరీ వినియోగం మరియు ఆలస్యాన్ని ప్రభావితం చేస్తుంది."
    ),
    "Tamil": (
        "பெரிய மொழி மாதிரிகள் உரையை டோக்கன்களாக செயலாக்குகின்றன. "
        "டோக்கனைசேஷன் சூழல் பயன்பாடு, அனுமான செலவு, நினைவக பயன்பாடு மற்றும் தாமதத்தை பாதிக்கிறது."
    ),
    "Malayalam": (
        "വലിയ ഭാഷാ മോഡലുകൾ ടെക്സ്റ്റിനെ ടോക്കണുകളായി പ്രോസസ്സ് ചെയ്യുന്നു. "
        "ടോക്കണൈസേഷൻ കോൺടെക്സ്റ്റ് ഉപയോഗം, ഇൻഫറൻസ് ചെലവ്, മെമ്മറി ഉപയോഗം, ലേറ്റൻസി എന്നിവയെ ബാധിക്കുന്നു."
    ),
    "Odia": (
        "ବଡ଼ ଭାଷା ମଡେଲଗୁଡ଼ିକ ପାଠ୍ୟକୁ ଟୋକେନ ଭାବରେ ପ୍ରକ୍ରିୟା କରନ୍ତି। "
        "ଟୋକେନାଇଜେସନ ପରିପ୍ରେକ୍ଷିତ ବ୍ୟବହାର, ଅନୁମାନ ଖର୍ଚ୍ଚ, ସ୍ମୃତି ବ୍ୟବହାର ଏବଂ ବିଳମ୍ବତାକୁ ପ୍ରଭାବିତ କରେ।"
    ),
}


for language, text in TEXTS.items():
    result = benchmark_text(tokenizer=tokenizer,text=text,language=language)
    print(result+"\n")