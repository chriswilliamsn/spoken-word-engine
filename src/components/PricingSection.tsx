import { Button } from "@/components/ui/button";
import { Check, Star } from "lucide-react";

const PricingSection = () => {
  const plans = [
    {
      name: "Basic",
      price: "9",
      description: "Perfect for individuals and small projects",
      features: [
        "10,000 characters/month",
        "3 voice models",
        "Basic customization",
        "MP3 downloads",
        "Email support"
      ],
      buttonVariant: "outline-glow" as const,
      popular: false
    },
    {
      name: "Professional", 
      price: "29",
      description: "Ideal for content creators and businesses",
      features: [
        "100,000 characters/month",
        "15 voice models",
        "Advanced customization",
        "Multiple formats (MP3, WAV)",
        "Priority support",
        "Commercial license",
        "API access"
      ],
      buttonVariant: "hero" as const,
      popular: true
    },
    {
      name: "Enterprise",
      price: "99",
      description: "For large organizations and agencies",
      features: [
        "Unlimited characters",
        "50+ voice models",
        "Full customization suite",
        "All formats + streaming",
        "24/7 phone support", 
        "Extended commercial license",
        "Advanced API & webhooks",
        "Custom voice training",
        "Team management"
      ],
      buttonVariant: "premium" as const,
      popular: false
    }
  ];

  return (
    <section id="pricing" className="py-24 bg-gradient-surface relative">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-6 text-foreground">
            Choose Your
            <span className="text-ai-primary"> Voice Plan</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Scale your voice generation needs with our flexible pricing options
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {plans.map((plan, index) => (
            <div 
              key={plan.name}
              className={`relative bg-card border rounded-2xl p-8 shadow-card hover:shadow-elevated transition-all duration-300 hover:scale-105 ${
                plan.popular 
                  ? 'border-ai-primary bg-gradient-to-b from-card to-ai-surface' 
                  : 'border-border hover:border-ai-primary/50'
              }`}
            >
              {plan.popular && (
                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                  <div className="bg-gradient-primary text-white px-4 py-1 rounded-full text-sm font-semibold flex items-center">
                    <Star className="w-4 h-4 mr-1" />
                    Most Popular
                  </div>
                </div>
              )}
              
              <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-foreground mb-2">{plan.name}</h3>
                <p className="text-muted-foreground mb-4">{plan.description}</p>
                <div className="flex items-baseline justify-center">
                  <span className="text-5xl font-bold text-ai-primary">${plan.price}</span>
                  <span className="text-muted-foreground ml-2">/month</span>
                </div>
              </div>

              <ul className="space-y-4 mb-8">
                {plan.features.map((feature, idx) => (
                  <li key={idx} className="flex items-center">
                    <Check className="w-5 h-5 text-ai-primary mr-3 flex-shrink-0" />
                    <span className="text-card-foreground">{feature}</span>
                  </li>
                ))}
              </ul>

              <Button 
                variant={plan.buttonVariant} 
                className="w-full"
                size="lg"
              >
                {plan.name === "Enterprise" ? "Contact Sales" : "Get Started"}
              </Button>
            </div>
          ))}
        </div>

        <div className="text-center mt-12">
          <p className="text-muted-foreground">
            All plans include a 14-day free trial • No credit card required
          </p>
        </div>
      </div>
    </section>
  );
};

export default PricingSection;