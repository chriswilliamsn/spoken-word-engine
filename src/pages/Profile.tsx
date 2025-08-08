import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";

const Profile = () => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [email, setEmail] = useState<string>("");
  const [userId, setUserId] = useState<string>("");
  const [displayName, setDisplayName] = useState<string>("");
  const [originalDisplayName, setOriginalDisplayName] = useState<string>("");

  useEffect(() => {
    document.title = "Profile - Flow Voice AI";
  }, []);

  useEffect(() => {
    let isMounted = true;

    const init = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (!session?.user) {
          window.location.href = "/auth";
          return;
        }

        const user = session.user;
        if (!isMounted) return;

        setEmail(user.email || "");
        setUserId(user.id);

        // Try to load profile display name
        const { data: profile } = await supabase
          .from("profiles")
          .select("display_name")
          .eq("user_id", user.id)
          .maybeSingle();

        const fallbackName = (user.user_metadata as any)?.display_name || (user.email?.split("@")[0] ?? "User");
        const name = profile?.display_name || fallbackName;
        if (!isMounted) return;
        setDisplayName(name);
        setOriginalDisplayName(name);
      } catch (e) {
        // ignore
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    init();
    return () => {
      isMounted = false;
    };
  }, []);

  const saveProfile = async () => {
    if (!userId) return;
    setLoading(true);
    try {
      // Check if profile exists
      const { data: existing } = await supabase
        .from("profiles")
        .select("id")
        .eq("user_id", userId)
        .maybeSingle();

      if (existing) {
        const { error } = await supabase
          .from("profiles")
          .update({ display_name: displayName })
          .eq("user_id", userId);
        if (error) throw error;
      } else {
        const { error } = await supabase
          .from("profiles")
          .insert({ user_id: userId, display_name: displayName });
        if (error) throw error;
      }

      setOriginalDisplayName(displayName);
      toast({ description: "Profile updated" });
    } catch (error: any) {
      toast({ description: error.message || "Failed to update profile", variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const signOut = async () => {
    await supabase.auth.signOut();
    window.location.href = "/";
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-muted-foreground">Loading profile...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background py-24 px-6">
      <div className="container mx-auto max-w-2xl">
        <Card className="bg-card/90 backdrop-blur-md border-border/50">
          <CardHeader>
            <CardTitle>Profile</CardTitle>
            <CardDescription>Manage your account details</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid gap-2">
              <Label>Email</Label>
              <Input value={email} disabled />
            </div>

            <div className="grid gap-2">
              <Label>Display name</Label>
              <Input
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="Your name"
              />
            </div>

            <div className="flex gap-3">
              <Button
                onClick={saveProfile}
                disabled={loading || displayName.trim() === "" || displayName === originalDisplayName}
              >
                Save changes
              </Button>
              <Button variant="outline" onClick={signOut}>Sign out</Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Profile;