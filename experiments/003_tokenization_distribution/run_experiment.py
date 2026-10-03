from statistics import mean, median

from llm_engineering_lab.tokenization.benchmark import (
    load_tokenizer,
    benchmark_text,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = load_tokenizer(MODEL_NAME)




TEXTS = {
    "English": [
        "Large language models process text as sequences of tokens.",
        "Tokenization affects memory usage, latency, and inference cost.",
        "Retrieval systems must carefully manage the available context window.",
        "Longer inputs require more computation during the prefill stage.",
        "Efficient tokenization can significantly reduce inference costs.",
    ],

    "Hindi": [
        "बड़े भाषा मॉडल पाठ को टोकन के अनुक्रम के रूप में संसाधित करते हैं।",
        "टोकनाइजेशन मेमोरी उपयोग, विलंबता और अनुमान लागत को प्रभावित करता है।",
        "रिट्रीवल सिस्टम को उपलब्ध संदर्भ विंडो का सावधानीपूर्वक प्रबंधन करना चाहिए।",
        "लंबे इनपुट के लिए प्रीफिल चरण के दौरान अधिक गणना की आवश्यकता होती है।",
        "कुशल टोकनाइजेशन अनुमान लागत को काफी कम कर सकता है।",
    ],

    "Telugu": [
        "పెద్ద భాషా నమూనాలు పాఠ్యాన్ని టోకెన్ల క్రమంగా ప్రాసెస్ చేస్తాయి.",
        "టోకెనైజేషన్ మెమరీ వినియోగం, ఆలస్యం మరియు అనుమితి ఖర్చును ప్రభావితం చేస్తుంది.",
        "రిట్రీవల్ వ్యవస్థలు అందుబాటులో ఉన్న సందర్భ విండోను జాగ్రత్తగా నిర్వహించాలి.",
        "పొడవైన ఇన్‌పుట్‌లకు ప్రీఫిల్ దశలో ఎక్కువ గణన అవసరం.",
        "సమర్థవంతమైన టోకెనైజేషన్ అనుమితి ఖర్చులను గణనీయంగా తగ్గిస్తుంది.",
    ],

    "Tamil": [
        "பெரிய மொழி மாதிரிகள் உரையை டோக்கன்களின் வரிசையாக செயலாக்குகின்றன.",
        "டோக்கனைசேஷன் நினைவக பயன்பாடு, தாமதம் மற்றும் அனுமான செலவை பாதிக்கிறது.",
        "மீட்டெடுப்பு அமைப்புகள் கிடைக்கக்கூடிய சூழல் சாளரத்தை கவனமாக நிர்வகிக்க வேண்டும்.",
        "நீண்ட உள்ளீடுகளுக்கு முன் நிரப்புதல் கட்டத்தில் அதிக கணக்கீடு தேவைப்படுகிறது.",
        "திறமையான டோக்கனைசேஷன் அனுமான செலவைக் கணிசமாகக் குறைக்கும்.",
    ],

    "Malayalam": [
        "വലിയ ഭാഷാ മോഡലുകൾ ടെക്സ്റ്റിനെ ടോക്കണുകളുടെ ക്രമമായി പ്രോസസ്സ് ചെയ്യുന്നു.",
        "ടോക്കനൈസേഷൻ മെമ്മറി ഉപയോഗം, ലേറ്റൻസി, ഇൻഫറൻസ് ചെലവ് എന്നിവയെ ബാധിക്കുന്നു.",
        "റിട്രീവൽ സംവിധാനങ്ങൾ ലഭ്യമായ കോൺടെക്സ്റ്റ് വിൻഡോ ശ്രദ്ധാപൂർവ്വം കൈകാര്യം ചെയ്യണം.",
        "ദൈർഘ്യമേറിയ ഇൻപുട്ടുകൾക്ക് പ്രീഫിൽ ഘട്ടത്തിൽ കൂടുതൽ കണക്കുകൂട്ടൽ ആവശ്യമാണ്.",
        "കാര്യക്ഷമമായ ടോക്കനൈസേഷൻ ഇൻഫറൻസ് ചെലവ് ഗണ്യമായി കുറയ്ക്കും.",
    ],

    "Odia": [
        "ବଡ଼ ଭାଷା ମଡେଲଗୁଡ଼ିକ ପାଠ୍ୟକୁ ଟୋକେନର କ୍ରମ ଭାବରେ ପ୍ରକ୍ରିୟା କରନ୍ତି।",
        "ଟୋକେନାଇଜେସନ ସ୍ମୃତି ବ୍ୟବହାର, ବିଳମ୍ବତା ଏବଂ ଅନୁମାନ ଖର୍ଚ୍ଚକୁ ପ୍ରଭାବିତ କରେ।",
        "ରିଟ୍ରିଭାଲ ସିଷ୍ଟମଗୁଡ଼ିକ ଉପଲବ୍ଧ ପରିପ୍ରେକ୍ଷିତ ୱିଣ୍ଡୋକୁ ସାବଧାନତାର ସହିତ ପରିଚାଳନା କରିବା ଉଚିତ।",
        "ଲମ୍ବା ଇନପୁଟ ପାଇଁ ପ୍ରିଫିଲ ପର୍ଯ୍ୟାୟରେ ଅଧିକ ଗଣନା ଆବଶ୍ୟକ ହୁଏ।",
        "ଦକ୍ଷ ଟୋକେନାଇଜେସନ ଅନୁମାନ ଖର୍ଚ୍ଚକୁ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ ଭାବେ କମାଇପାରେ।",
    ],
}


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0

    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values))

    if lower == index:
        return values[lower]

    return values[lower] + (values[upper] - values[lower]) * (
        index - lower
    )


for language, sentences in TEXTS.items():

    ratios = []
    token_counts = []

    for sentence in sentences:
        result = benchmark_text(tokenizer, sentence,language)

        ratios.append(result["tokens_per_character"])
        token_counts.append(result["tokens"])

    print(f"\n{'=' * 70}")
    print(language)
    print("=" * 70)

    print(f"Sentences: {len(sentences)}")
    print(f"Mean tokens: {mean(token_counts):.2f}")
    print(f"Median tokens: {median(token_counts):.2f}")

    print(f"Mean tokens/character: {mean(ratios):.3f}")
    print(f"Median tokens/character: {median(ratios):.3f}")
    print(f"Min tokens/character: {min(ratios):.3f}")
    print(f"Max tokens/character: {max(ratios):.3f}")
    print(f"P95 tokens/character: {percentile(ratios, 0.95):.3f}")