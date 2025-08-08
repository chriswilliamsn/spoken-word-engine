import { Button } from "@/components/ui/button";
import { Play, Volume2, Square } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import heroBackground from "@/assets/hero-background.jpg";
import { useRef, useState } from "react";
import { supabase } from "@/integrations/supabase/client";

const HeroSection = () => {
  const { toast } = useToast();
  const textRef = useRef<HTMLTextAreaElement>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const speechUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  const stopSpeech = () => {
    if (speechUtteranceRef.current && speechSynthesis.speaking) {
      speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  };

  const handlePreview = async () => {
    const textToRead = textRef.current?.value || "Welcome to VoiceAI, where cutting-edge technology meets natural human expression.";
    
    // Stop any current speech
    stopSpeech();

    // Start speech synthesis and show toast with stop button
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(textToRead);
      speechUtteranceRef.current = utterance;
      
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);
      
      speechSynthesis.speak(utterance);
      setIsSpeaking(true); // Set immediately for the toast
    }

    // Generate and play audio with Nari (Dia TTS). Falls back silently if unavailable.
    try {
      const { data, error } = await supabase.functions.invoke('dia-tts', {
        body: {
          text: textToRead,
          max_tokens: 3072,
          temperature: 0.7,
          top_p: 0.9,
        },
      });
      if (!error && data?.audio_content) {
        const audioBlob = new Blob(
          [Uint8Array.from(atob(data.audio_content), c => c.charCodeAt(0))],
          { type: 'audio/wav' }
        );
        const url = URL.createObjectURL(audioBlob);
        const audio = new Audio(url);
        audio.play();
      }
    } catch (_) {
      // no-op if TTS generation fails
    }

    toast({
      description: textToRead,
      duration: 10000,
      action: (
        <Button variant="outline" size="sm" onClick={stopSpeech}>
          <Square className="h-4 w-4 mr-1" />
          Stop
        </Button>
      ),
    });
  };
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background Image */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat opacity-20"
        style={{ backgroundImage: `url(${heroBackground})` }}
      />
      
      {/* Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-hero" />
      
      {/* Content */}
      <div className="relative z-10 container mx-auto px-6 text-center">
        <div className="max-w-4xl mx-auto animate-slide-up">
          <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-white via-ai-primary to-ai-secondary bg-clip-text text-transparent leading-tight">
            Transform Text Into
            <span className="block text-ai-primary animate-glow-pulse">Natural Speech</span>
          </h1>
          
          <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
            Powered by Nari Labs advanced AI, create professional-grade voiceovers 
            with human-like emotion and clarity in seconds.
          </p>
          
          {/* Demo Section */}
          <div className="bg-card/40 backdrop-blur-sm border border-border/50 rounded-2xl p-8 mb-8 max-w-2xl mx-auto shadow-card">
            <h3 className="text-lg font-semibold mb-4 text-foreground">Try it now:</h3>
            <div className="bg-ai-surface rounded-lg p-4 mb-4">
              <textarea 
                ref={textRef}
                className="w-full bg-transparent text-foreground placeholder-muted-foreground resize-none border-none outline-none"
                rows={3}
                placeholder="Enter your text here to convert to speech..."
                defaultValue="Welcome to VoiceAI, where cutting-edge technology meets natural human expression."
              />
            </div>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button variant="hero" className="flex-1 group" onClick={() => window.location.href = '/auth'}>
                <Play className="mr-2 group-hover:scale-110 transition-transform" />
                Generate Speech
              </Button>
              <Button variant="outline-glow" className="flex-1" onClick={handlePreview}>
                <Volume2 className="mr-2" />
                Preview
              </Button>
            </div>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button variant="hero" size="lg" className="animate-float" onClick={() => window.location.href = '/auth'}>
              Start Free Trial
            </Button>
            <Button variant="outline-glow" size="lg">
              View Pricing
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;