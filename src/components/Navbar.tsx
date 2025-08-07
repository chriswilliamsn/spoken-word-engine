import { Button } from "@/components/ui/button";

const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b border-border">
      <div className="container mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-gradient-primary rounded-lg"></div>
          <span className="text-xl font-bold text-foreground">Flow Voice AI</span>
        </div>
        
        <div className="hidden md:flex items-center space-x-8">
          <a href="/features" className="text-muted-foreground hover:text-foreground transition-colors">
            Features
          </a>
          <a href="#pricing" className="text-muted-foreground hover:text-foreground transition-colors">
            Pricing
          </a>
          <a href="/about" className="text-muted-foreground hover:text-foreground transition-colors">
            About
          </a>
        </div>

        <div className="flex items-center space-x-4">
          <Button variant="ghost" size="sm" onClick={() => window.location.href = '/auth'}>
            Sign In
          </Button>
          <Button variant="hero" size="sm" onClick={() => window.location.href = '/auth'}>
            Get Started
          </Button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;