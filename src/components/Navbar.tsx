import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { supabase } from "@/integrations/supabase/client";

const Navbar = () => {
  const [displayName, setDisplayName] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    const load = async () => {
      // Listen for auth changes
      const { data: { subscription } } = supabase.auth.onAuthStateChange((_, session) => {
        if (!isMounted) return;
        const user = session?.user || null;
        if (user) {
          const fallback = (user.user_metadata as any)?.display_name || user.email?.split("@")[0] || "Account";
          // Defer profile fetch to avoid blocking callback
          setTimeout(async () => {
            try {
              const { data: profile } = await supabase
                .from("profiles")
                .select("display_name")
                .eq("user_id", user.id)
                .maybeSingle();
              if (!isMounted) return;
              setDisplayName(profile?.display_name || fallback);
            } catch {
              if (!isMounted) return;
              setDisplayName(fallback);
            }
          }, 0);
        } else {
          setDisplayName(null);
        }
      });

      // Initialize with current session
      const { data: { session } } = await supabase.auth.getSession();
      const user = session?.user || null;
      if (user) {
        const fallback = (user.user_metadata as any)?.display_name || user.email?.split("@")[0] || "Account";
        try {
          const { data: profile } = await supabase
            .from("profiles")
            .select("display_name")
            .eq("user_id", user.id)
            .maybeSingle();
          if (!isMounted) return;
          setDisplayName(profile?.display_name || fallback);
        } catch {
          if (!isMounted) return;
          setDisplayName(fallback);
        }
      }

      return () => subscription.unsubscribe();
    };

    const cleanup = load();
    return () => {
      isMounted = false;
      // Ensure subscription cleanup if load resolved
      Promise.resolve(cleanup).catch(() => {});
    };
  }, []);

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
          <a href="/tts" className="text-muted-foreground hover:text-foreground transition-colors">
            TTS Demo
          </a>
          <a href="#pricing" className="text-muted-foreground hover:text-foreground transition-colors">
            Pricing
          </a>
          <a href="/about" className="text-muted-foreground hover:text-foreground transition-colors">
            About
          </a>
        </div>

        <div className="flex items-center space-x-4">
          {displayName ? (
            <Button variant="ghost" size="sm" onClick={() => (window.location.href = "/profile")}> 
              {displayName}
            </Button>
          ) : (
            <Button variant="ghost" size="sm" onClick={() => (window.location.href = "/auth")}> 
              Sign In
            </Button>
          )}
          <Button variant="hero" size="sm" onClick={() => (window.location.href = displayName ? "/tts" : "/auth")}>
            {displayName ? "Create" : "Get Started"}
          </Button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;