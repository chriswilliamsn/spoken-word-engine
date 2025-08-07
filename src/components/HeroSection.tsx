import { Button } from "@/components/ui/button";
import { Play, Volume2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import heroBackground from "@/assets/hero-background.jpg";

const HeroSection = () => {
  const { toast } = useToast();

  const handlePreview = () => {
    toast({
      title: "Welcome Preview for Flow Voice Text-to-Speech",
      description: "Welcome to Flow Voice!\n\nExperience the future of text-to-speech technology with Flow Voice, the leading realistic voice model on the market. Our state-of-the-art system uses advanced AI algorithms to deliver natural-sounding speech that captures the nuances of human emotion and tone.\n\nKey Features:\n• High Fidelity Sound: Enjoy crystal-clear audio that makes every word resonate.\n• Natural Intonation: Flow Voice mimics the rhythm and inflection of human speech, making your content engaging and relatable.\n• Customizable Voices: Choose from a diverse range of voices and accents to suit your needs.\n• User-Friendly Interface: Effortlessly convert text to speech with our intuitive platform.\n\nApplications:\n• E-Learning: Enhance your educational content with lifelike narration.\n• Audiobooks: Bring your stories to life with expressive reading.\n• Accessibility: Provide a voice for those who need assistance with reading.\n\nJoin the revolution in voice technology and bring your text to life with Flow Voice. Start your journey today and experience the difference!",
      duration: 10000,
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