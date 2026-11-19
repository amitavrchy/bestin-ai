import edge_tts

VOICES = [
    {"id": i, "name": n, "gender": g, "locale": loc, "provider": "edge"}
    for i, n, g, loc in [
        ("en-US-AriaNeural", "Aria", "Female", "en-US"),
        ("en-US-JennyNeural", "Jenny", "Female", "en-US"),
        ("en-US-GuyNeural", "Guy", "Male", "en-US"),
        ("en-US-ChristopherNeural", "Christopher", "Male", "en-US"),
        ("en-GB-SoniaNeural", "Sonia", "Female", "en-GB"),
        ("en-GB-RyanNeural", "Ryan", "Male", "en-GB"),
        ("en-AU-NatashaNeural", "Natasha", "Female", "en-AU"),
        ("en-AU-WilliamNeural", "William", "Male", "en-AU"),
        ("en-IN-NeerjaNeural", "Neerja", "Female", "en-IN"),
        ("en-IN-PrabhatNeural", "Prabhat", "Male", "en-IN"),
        ("bn-BD-NabanitaNeural", "Nabanita", "Female", "bn-BD"),
        ("hi-IN-SwaraNeural", "Swara", "Female", "hi-IN"),
        ("fr-FR-DeniseNeural", "Denise", "Female", "fr-FR"),
        ("es-ES-ElviraNeural", "Elvira", "Female", "es-ES"),
    ]
]


async def synthesize(text: str, voice: str) -> tuple[bytes, list[dict]]:
    audio_parts = []
    words = []
    async for item in edge_tts.Communicate(text, voice, boundary="WordBoundary").stream():
        if item["type"] == "audio":
            audio_parts.append(item["data"])
        elif item["type"] == "WordBoundary":
            words.append({"word": item["text"], "start": item["offset"] / 10_000_000, "end": (item["offset"] + item["duration"]) / 10_000_000})
    return b"".join(audio_parts), words
