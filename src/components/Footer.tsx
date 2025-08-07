const Footer = () => {
  return (
    <footer className="bg-ai-surface border-t border-border py-12">
      <div className="container mx-auto px-6">
        <div className="grid md:grid-cols-4 gap-8">
          <div className="md:col-span-2">
            <div className="flex items-center space-x-2 mb-4">
              <div className="w-8 h-8 bg-gradient-primary rounded-lg"></div>
              <span className="text-xl font-bold text-foreground">VoiceAI</span>
            </div>
            <p className="text-muted-foreground mb-4 max-w-md">
              Transform your text into natural, human-like speech with our advanced AI technology powered by Nari Labs.
            </p>
            <div className="text-sm text-muted-foreground">
              © 2024 VoiceAI. All rights reserved.
            </div>
          </div>
          
          <div>
            <h4 className="font-semibold text-foreground mb-4">Product</h4>
            <ul className="space-y-2">
              <li><a href="#" className="text-muted-foreground hover:text-ai-primary transition-colors">Features</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-ai-primary transition-colors">Pricing</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-ai-primary transition-colors">API</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-ai-primary transition-colors">Documentation</a></li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-semibold text-foreground mb-4">Support</h4>
            <ul className="space-y-2">
              <li><a href="#" className="text-muted-foreground hover:text-ai-primary transition-colors">Help Center</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-ai-primary transition-colors">Contact</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-ai-primary transition-colors">Status</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-ai-primary transition-colors">Privacy</a></li>
            </ul>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;