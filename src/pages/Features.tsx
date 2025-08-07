import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Mic, 
  Zap, 
  Globe, 
  Volume2, 
  Users, 
  Shield, 
  Clock, 
  Cpu, 
  FileAudio, 
  Languages,
  Download,
  Settings
} from "lucide-react";

const Features = () => {
  const features = [
    {
      icon: <Mic className="h-8 w-8" />,
      title: "Natural Voice Synthesis",
      description: "Advanced AI models that generate human-like speech with natural intonation and emotion.",
      highlights: ["Neural voice cloning", "Emotion recognition", "Prosody control"]
    },
    {
      icon: <Languages className="h-8 w-8" />,
      title: "Multi-Language Support",
      description: "Support for 50+ languages and dialects with native speaker quality pronunciation.",
      highlights: ["Real-time translation", "Accent adaptation", "Cultural context"]
    },
    {
      icon: <Zap className="h-8 w-8" />,
      title: "Real-Time Processing",
      description: "Lightning-fast speech generation with minimal latency for live applications.",
      highlights: ["Sub-second response", "Streaming audio", "Edge optimization"]
    },
    {
      icon: <Volume2 className="h-8 w-8" />,
      title: "Voice Customization",
      description: "Fine-tune voice characteristics including pitch, speed, and emotional tone.",
      highlights: ["Custom voice profiles", "Tone adjustment", "Speed control"]
    },
    {
      icon: <FileAudio className="h-8 w-8" />,
      title: "Multiple Audio Formats",
      description: "Export in various audio formats optimized for different use cases and platforms.",
      highlights: ["MP3, WAV, OGG", "Quality optimization", "Compression options"]
    },
    {
      icon: <Cpu className="h-8 w-8" />,
      title: "API Integration",
      description: "Robust APIs with comprehensive SDKs for seamless integration into your applications.",
      highlights: ["RESTful APIs", "WebSocket support", "SDK libraries"]
    },
    {
      icon: <Users className="h-8 w-8" />,
      title: "Batch Processing",
      description: "Process large volumes of text efficiently with our scalable batch processing system.",
      highlights: ["Queue management", "Progress tracking", "Auto-scaling"]
    },
    {
      icon: <Shield className="h-8 w-8" />,
      title: "Enterprise Security",
      description: "Bank-grade security with encryption, compliance, and data privacy protection.",
      highlights: ["End-to-end encryption", "GDPR compliant", "SOC 2 certified"]
    },
    {
      icon: <Clock className="h-8 w-8" />,
      title: "24/7 Availability",
      description: "Reliable service with 99.9% uptime guarantee and global infrastructure.",
      highlights: ["Global CDN", "Auto-failover", "Performance monitoring"]
    }
  ];

  const useCases = [
    {
      title: "Content Creation",
      description: "Transform written content into engaging audio for podcasts, audiobooks, and videos.",
      icon: <FileAudio className="h-6 w-6" />
    },
    {
      title: "Accessibility",
      description: "Make digital content accessible with screen reader compatibility and audio alternatives.",
      icon: <Users className="h-6 w-6" />
    },
    {
      title: "E-Learning",
      description: "Create immersive educational experiences with natural-sounding narration.",
      icon: <Globe className="h-6 w-6" />
    },
    {
      title: "Customer Service",
      description: "Enhance IVR systems and chatbots with human-like voice interactions.",
      icon: <Settings className="h-6 w-6" />
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="pt-24 pb-16 px-4">
        <div className="max-w-6xl mx-auto text-center">
          <Badge variant="secondary" className="mb-4">
            Powered by Nari Labs
          </Badge>
          <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-primary bg-clip-text text-transparent">
            Advanced Features
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-8">
            Discover the cutting-edge capabilities that make our text-to-speech platform 
            the choice for developers, creators, and enterprises worldwide.
          </p>
          <Button variant="hero" size="lg" className="mr-4">
            Try Free Demo
          </Button>
          <Button variant="outline-glow" size="lg">
            View Documentation
          </Button>
        </div>
      </section>

      {/* Main Features Grid */}
      <section className="py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Powerful Features
            </h2>
            <p className="text-xl text-muted-foreground">
              Everything you need to create exceptional voice experiences
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <Card key={index} className="bg-card/50 backdrop-blur-sm border-border/50 hover:border-ai-primary/50 transition-all duration-300 hover:shadow-card">
                <CardHeader>
                  <div className="text-ai-primary mb-2">
                    {feature.icon}
                  </div>
                  <CardTitle className="text-xl">{feature.title}</CardTitle>
                  <CardDescription className="text-muted-foreground">
                    {feature.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {feature.highlights.map((highlight, idx) => (
                      <div key={idx} className="flex items-center text-sm text-muted-foreground">
                        <div className="w-1.5 h-1.5 bg-ai-primary rounded-full mr-2" />
                        {highlight}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-16 px-4 bg-card/20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Use Cases
            </h2>
            <p className="text-xl text-muted-foreground">
              See how our technology transforms industries
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {useCases.map((useCase, index) => (
              <Card key={index} className="text-center bg-card/50 backdrop-blur-sm border-border/50 hover:border-ai-primary/50 transition-all duration-300">
                <CardHeader>
                  <div className="text-ai-primary mx-auto mb-2">
                    {useCase.icon}
                  </div>
                  <CardTitle className="text-lg">{useCase.title}</CardTitle>
                  <CardDescription className="text-sm">
                    {useCase.description}
                  </CardDescription>
                </CardHeader>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Technical Specifications */}
      <section className="py-16 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Technical Specifications
            </h2>
            <p className="text-xl text-muted-foreground">
              Built for performance and reliability
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8">
            <Card className="bg-card/50 backdrop-blur-sm border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Zap className="h-5 w-5 text-ai-primary mr-2" />
                  Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Latency</span>
                  <span className="font-medium">&lt; 500ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Throughput</span>
                  <span className="font-medium">1000+ req/sec</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Uptime</span>
                  <span className="font-medium">99.9%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Audio Quality</span>
                  <span className="font-medium">48kHz/16-bit</span>
                </div>
              </CardContent>
            </Card>
            
            <Card className="bg-card/50 backdrop-blur-sm border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Globe className="h-5 w-5 text-ai-primary mr-2" />
                  Coverage
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Languages</span>
                  <span className="font-medium">50+</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Voice Models</span>
                  <span className="font-medium">200+</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Regions</span>
                  <span className="font-medium">Global</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Data Centers</span>
                  <span className="font-medium">15+</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-4 bg-gradient-primary">
        <div className="max-w-4xl mx-auto text-center text-white">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Ready to Get Started?
          </h2>
          <p className="text-xl opacity-90 mb-8">
            Experience the power of advanced text-to-speech technology today
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="secondary" size="lg">
              Start Free Trial
            </Button>
            <Button variant="outline" size="lg" className="border-white text-white hover:bg-white hover:text-primary">
              Contact Sales
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Features;