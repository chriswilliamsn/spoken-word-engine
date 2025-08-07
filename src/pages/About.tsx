import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Target, 
  Users, 
  Award, 
  Globe, 
  Heart, 
  Lightbulb, 
  Shield, 
  Rocket,
  Github,
  Linkedin,
  Twitter
} from "lucide-react";

const About = () => {
  const team = [
    {
      name: "Dr. Sarah Chen",
      role: "CEO & Co-founder",
      image: "/placeholder.svg",
      description: "Former Google AI researcher with 10+ years in speech synthesis and natural language processing."
    },
    {
      name: "Michael Rodriguez",
      role: "CTO & Co-founder", 
      image: "/placeholder.svg",
      description: "Ex-Amazon Alexa engineer specializing in real-time audio processing and distributed systems."
    },
    {
      name: "Emily Watson",
      role: "Head of AI Research",
      image: "/placeholder.svg", 
      description: "PhD in Computational Linguistics from Stanford, leading voice cloning and emotion synthesis research."
    },
    {
      name: "David Kim",
      role: "VP of Engineering",
      image: "/placeholder.svg",
      description: "Previously at Spotify, expert in audio infrastructure and high-performance computing systems."
    }
  ];

  const values = [
    {
      icon: <Lightbulb className="h-6 w-6" />,
      title: "Innovation",
      description: "Pushing the boundaries of what's possible with AI-powered voice technology."
    },
    {
      icon: <Users className="h-6 w-6" />,
      title: "Accessibility",
      description: "Making digital content accessible to everyone, regardless of their abilities."
    },
    {
      icon: <Shield className="h-6 w-6" />,
      title: "Privacy",
      description: "Protecting user data with enterprise-grade security and transparent practices."
    },
    {
      icon: <Heart className="h-6 w-6" />,
      title: "Quality",
      description: "Delivering the highest quality voice synthesis with human-like naturalness."
    }
  ];

  const milestones = [
    {
      year: "2020",
      title: "Company Founded",
      description: "Nari Labs was established with a vision to democratize voice technology."
    },
    {
      year: "2021", 
      title: "First AI Model",
      description: "Released our breakthrough neural voice synthesis model with 50+ languages."
    },
    {
      year: "2022",
      title: "Series A Funding",
      description: "Raised $15M to accelerate research and expand our global infrastructure."
    },
    {
      year: "2023",
      title: "Enterprise Launch",
      description: "Launched enterprise solutions serving Fortune 500 companies worldwide."
    },
    {
      year: "2024",
      title: "Real-time Processing",
      description: "Achieved sub-500ms latency with our next-generation processing architecture."
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="pt-24 pb-16 px-4">
        <div className="max-w-6xl mx-auto text-center">
          <Badge variant="secondary" className="mb-4">
            About Nari Labs
          </Badge>
          <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-primary bg-clip-text text-transparent">
            Pioneering the Future of Voice
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-8">
            We're building the next generation of text-to-speech technology that sounds 
            indistinguishable from human speech, making digital content more accessible 
            and engaging for everyone.
          </p>
        </div>
      </section>

      {/* Mission & Vision */}
      <section className="py-16 px-4 bg-card/20">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12">
            <Card className="bg-card/50 backdrop-blur-sm border-border/50">
              <CardHeader>
                <div className="text-ai-primary mb-2">
                  <Target className="h-8 w-8" />
                </div>
                <CardTitle className="text-2xl">Our Mission</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground text-lg leading-relaxed">
                  To democratize access to high-quality voice technology, enabling creators, 
                  developers, and businesses to build more inclusive and engaging digital experiences 
                  that break down barriers between text and speech.
                </p>
              </CardContent>
            </Card>

            <Card className="bg-card/50 backdrop-blur-sm border-border/50">
              <CardHeader>
                <div className="text-ai-primary mb-2">
                  <Rocket className="h-8 w-8" />
                </div>
                <CardTitle className="text-2xl">Our Vision</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground text-lg leading-relaxed">
                  A world where every piece of written content can be instantly transformed into 
                  natural, expressive speech in any language, making information accessible to 
                  billions of people regardless of literacy, visual ability, or language barriers.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Company Story */}
      <section className="py-16 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Our Story</h2>
            <p className="text-xl text-muted-foreground">
              From research lab to global platform
            </p>
          </div>
          
          <div className="prose prose-lg max-w-none text-muted-foreground">
            <p className="text-lg leading-relaxed mb-6">
              Nari Labs was born from a simple observation: despite decades of progress in AI, 
              most text-to-speech systems still sounded robotic and unnatural. Our founders, 
              Dr. Sarah Chen and Michael Rodriguez, met while working on voice AI projects at 
              leading tech companies and realized they shared a vision for something better.
            </p>
            <p className="text-lg leading-relaxed mb-6">
              Starting in a small research lab in 2020, we set out to solve the fundamental 
              challenges of natural speech synthesis. Our breakthrough came with the development 
              of a novel neural architecture that could capture the subtle nuances of human speech 
              – the rhythm, emotion, and natural flow that makes conversation engaging.
            </p>
            <p className="text-lg leading-relaxed">
              Today, our technology powers voice experiences for millions of users worldwide, 
              from audiobook narration to accessibility tools, from customer service to creative 
              content. We're proud to be at the forefront of making the digital world more accessible 
              and inclusive through the power of natural voice synthesis.
            </p>
          </div>
        </div>
      </section>

      {/* Values */}
      <section className="py-16 px-4 bg-card/20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Our Values</h2>
            <p className="text-xl text-muted-foreground">
              The principles that guide everything we do
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {values.map((value, index) => (
              <Card key={index} className="text-center bg-card/50 backdrop-blur-sm border-border/50 hover:border-ai-primary/50 transition-all duration-300">
                <CardHeader>
                  <div className="text-ai-primary mx-auto mb-2">
                    {value.icon}
                  </div>
                  <CardTitle className="text-lg">{value.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">{value.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Timeline */}
      <section className="py-16 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Our Journey</h2>
            <p className="text-xl text-muted-foreground">
              Key milestones in our mission to transform voice technology
            </p>
          </div>
          
          <div className="space-y-8">
            {milestones.map((milestone, index) => (
              <div key={index} className="flex items-start space-x-4">
                <div className="flex-shrink-0 w-16 h-16 bg-gradient-primary rounded-full flex items-center justify-center text-white font-bold">
                  {milestone.year}
                </div>
                <div className="flex-grow">
                  <h3 className="text-xl font-semibold mb-2">{milestone.title}</h3>
                  <p className="text-muted-foreground">{milestone.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="py-16 px-4 bg-card/20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Meet Our Team</h2>
            <p className="text-xl text-muted-foreground">
              The experts behind breakthrough voice technology
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {team.map((member, index) => (
              <Card key={index} className="text-center bg-card/50 backdrop-blur-sm border-border/50">
                <CardHeader>
                  <div className="w-24 h-24 bg-gradient-primary rounded-full mx-auto mb-4"></div>
                  <CardTitle className="text-lg">{member.name}</CardTitle>
                  <CardDescription className="text-ai-primary font-medium">
                    {member.role}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">{member.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold text-ai-primary mb-2">50M+</div>
              <div className="text-muted-foreground">Audio Generated</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-ai-primary mb-2">50+</div>
              <div className="text-muted-foreground">Languages</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-ai-primary mb-2">99.9%</div>
              <div className="text-muted-foreground">Uptime</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-ai-primary mb-2">1000+</div>
              <div className="text-muted-foreground">Enterprise Clients</div>
            </div>
          </div>
        </div>
      </section>

      {/* Contact CTA */}
      <section className="py-16 px-4 bg-gradient-primary">
        <div className="max-w-4xl mx-auto text-center text-white">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Join Us on This Journey
          </h2>
          <p className="text-xl opacity-90 mb-8">
            Whether you're a developer, creator, or enterprise, we'd love to hear from you
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="secondary" size="lg">
              Contact Us
            </Button>
            <Button variant="outline" size="lg" className="border-white text-white hover:bg-white hover:text-primary">
              Join Our Team
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default About;