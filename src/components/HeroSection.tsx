import { Button } from "@/components/ui/button";
import { Play, Volume2 } from "lucide-react";
import heroBackground from "@/assets/hero-background.jpg";

const HeroSection = () => {
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
              <Button variant="hero" className="flex-1 group">
                <Play className="mr-2 group-hover:scale-110 transition-transform" />
                Generate Speech
              </Button>
              <Button variant="outline-glow" className="flex-1">
                <Volume2 className="mr-2" />
                Preview
              </Button>
            </div>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button variant="hero" size="lg" className="animate-float">
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