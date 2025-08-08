import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

const Profile = () => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [email, setEmail] = useState<string>("");
  const [userId, setUserId] = useState<string>("");
  const [displayName, setDisplayName] = useState<string>("");
  const [originalDisplayName, setOriginalDisplayName] = useState<string>("");
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);

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

        // Try to load profile
        const { data: profile } = await supabase
          .from("profiles")
          .select("display_name, avatar_url")
          .eq("user_id", user.id)
          .maybeSingle();

        const fallbackName = (user.user_metadata as any)?.display_name || (user.email?.split("@")[0] ?? "User");
        const name = profile?.display_name || fallbackName;
        if (!isMounted) return;
        setDisplayName(name);
        setOriginalDisplayName(name);
        setAvatarUrl(profile?.avatar_url ?? null);
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
        <Card className="bg-card/90 backdrop-blur-md border-border/50 mb-6">
          <CardHeader>
            <CardTitle>Profile</CardTitle>
            <CardDescription>Manage your account details</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Avatar + Upload */}
            <div className="flex items-center gap-4">
              <Avatar className="h-16 w-16 shadow-sm">
                <AvatarImage src={avatarUrl || undefined} alt={displayName || email} />
                <AvatarFallback>{(displayName || email || "U").slice(0,2).toUpperCase()}</AvatarFallback>
              </Avatar>
              <div className="space-y-2">
                <div className="text-sm text-muted-foreground">Profile picture</div>
                <div className="flex items-center gap-3">
                  <label className="inline-flex">
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={async (e) => {
                        const file = e.target.files?.[0];
                        if (!file || !userId) return;
                        try {
                          setUploading(true);
                          const filePath = `${userId}/${Date.now()}-${file.name}`;
                          const { error: uploadError } = await supabase.storage
                            .from('avatars')
                            .upload(filePath, file, { upsert: true, contentType: file.type });
                          if (uploadError) throw uploadError;
                          const { data: pub } = supabase.storage.from('avatars').getPublicUrl(filePath);
                          const newUrl = pub.publicUrl;
                          // Try update first
                          const { data: updated, error: updErr } = await supabase
                            .from('profiles')
                            .update({ avatar_url: newUrl })
                            .eq('user_id', userId)
                            .select('id');
                          if (updErr) throw updErr;
                          if (!updated || updated.length === 0) {
                            const fallbackName = displayName || (email?.split('@')[0] ?? 'User');
                            const { error: insErr } = await supabase
                              .from('profiles')
                              .insert({ user_id: userId, avatar_url: newUrl, display_name: fallbackName });
                            if (insErr) throw insErr;
                          }
                          setAvatarUrl(newUrl);
                          toast({ description: 'Avatar updated' });
                        } catch (err: any) {
                          toast({ description: err.message || 'Failed to upload avatar', variant: 'destructive' });
                        } finally {
                          setUploading(false);
                          // reset input value so same file can be reselected
                          e.currentTarget.value = '';
                        }
                      }}
                    />
                    <span className="px-4 py-2 rounded-md bg-muted hover:bg-muted/80 cursor-pointer text-sm">
                      {uploading ? 'Uploading…' : 'Upload new'}
                    </span>
                  </label>
                  {avatarUrl && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={async () => {
                        setAvatarUrl(null);
                        try {
                          const { error } = await supabase
                            .from('profiles')
                            .update({ avatar_url: null })
                            .eq('user_id', userId);
                          if (error) throw error;
                          toast({ description: 'Avatar removed' });
                        } catch (err: any) {
                          toast({ description: err.message || 'Failed to remove avatar', variant: 'destructive' });
                        }
                      }}
                    >
                      Remove
                    </Button>
                  )}
                </div>
              </div>
            </div>

            {/* Email */}
            <div className="grid gap-2">
              <Label>Email</Label>
              <Input value={email} disabled />
            </div>

            {/* Display Name */}
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

        {/* Subscription card */}
        <Card className="bg-card/90 backdrop-blur-md border-border/50">
          <CardHeader>
            <CardTitle>Subscription</CardTitle>
            <CardDescription>Your plan and billing</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-muted-foreground">Status</div>
                <div className="text-foreground font-medium">Not active</div>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" onClick={() => (window.location.href = '/#pricing')}>View Plans</Button>
                <Button disabled variant="secondary">Manage</Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Profile;