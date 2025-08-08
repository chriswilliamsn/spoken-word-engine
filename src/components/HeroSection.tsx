import { Button } from "@/components/ui/button";
import { Play, Volume2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import heroBackground from "@/assets/hero-background.jpg";
import { useRef } from "react";
import { supabase } from "@/integrations/supabase/client";
import { ToastAction } from "@/components/ui/toast";

const HeroSection = () => {
  const { toast } = useToast();
  const textRef = useRef<HTMLTextAreaElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  const handlePreview = async () => {
    const previewText =
      "Welcome to Flow Voice — a quick preview of our natural, expressive TTS. Hear a short sample now.";

    // Stop any ongoing playback first
    if (audioRef.current) {
      try { audioRef.current.pause(); } catch {}
      audioRef.current = null;
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try { window.speechSynthesis.cancel(); } catch {}
      utteranceRef.current = null;
    }

    let dismissToast: () => void = () => {};
    const stopPlayback = () => {
      try { audioRef.current?.pause(); } catch {}
      audioRef.current = null;
      try { if (typeof window !== 'undefined' && 'speechSynthesis' in window) window.speechSynthesis.cancel(); } catch {}
      utteranceRef.current = null;
      dismissToast();
    };

    // Create a persistent toast with a Stop action; we'll dismiss when audio ends
    ({ dismiss: dismissToast } = toast({
      description: previewText,
      duration: 3600000,
      action: (
        <ToastAction altText="Stop playback" onClick={stopPlayback}>
          Stop
        </ToastAction>
      ),
    }));

    let spoke = false;

    // 1) Try Nari (Dia TTS)
    try {
      const { data, error } = await supabase.functions.invoke('dia-tts', {
        body: {
          text: previewText,
          max_tokens: 3072,
          temperature: 0.7,
          top_p: 0.9,
        },
      });
      if (!error && data?.audio_content) {
        const audioBlob = new Blob(
          [Uint8Array.from(atob(data.audio_content), (c) => c.charCodeAt(0))],
          { type: 'audio/wav' }
        );
        const url = URL.createObjectURL(audioBlob);
        const audio = new Audio(url);
        audioRef.current = audio;
        audio.onended = () => dismissToast();
        await audio.play();
        spoke = true;
      }
    } catch (_) {
      // continue to fallback
    }

    // 2) Fallback to OpenAI TTS if Nari unavailable
    if (!spoke) {
      try {
        const { data, error } = await supabase.functions.invoke('text-to-speech', {
          body: {
            text: previewText,
            voice: 'alloy',
            speed: 1,
          },
        });
        if (!error && data?.audioContent) {
          const audioBlob = new Blob(
            [Uint8Array.from(atob(data.audioContent), (c) => c.charCodeAt(0))],
            { type: 'audio/mpeg' }
          );
          const url = URL.createObjectURL(audioBlob);
          const audio = new Audio(url);
          audioRef.current = audio;
          audio.onended = () => dismissToast();
          await audio.play();
          spoke = true;
        }
      } catch (_) {
        // continue to browser TTS
      }
    }

    // 3) Final fallback: Browser SpeechSynthesis
    if (!spoke && typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        const synth = window.speechSynthesis;
        synth.cancel();
        const utterance = new SpeechSynthesisUtterance(previewText);
        utteranceRef.current = utterance;
        utterance.rate = 1;
        utterance.pitch = 1;
        utterance.lang = 'en-US';
        utterance.onend = () => dismissToast();
        synth.speak(utterance);
        spoke = true;
      } catch (_) {
        // swallow
      }
    }
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