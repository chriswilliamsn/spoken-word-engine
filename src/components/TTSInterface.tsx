import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Play, Pause, Download, Volume2, Loader2 } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';

const voices = [
  { id: 'alloy', name: 'Alloy', description: 'Neutral, balanced voice' },
  { id: 'echo', name: 'Echo', description: 'Clear, articulate voice' },
  { id: 'fable', name: 'Fable', description: 'Warm, expressive voice' },
  { id: 'onyx', name: 'Onyx', description: 'Deep, rich voice' },
  { id: 'nova', name: 'Nova', description: 'Bright, energetic voice' },
  { id: 'shimmer', name: 'Shimmer', description: 'Soft, gentle voice' },
];

export default function TTSInterface() {
  const [text, setText] = useState('');
  const [selectedVoice, setSelectedVoice] = useState('alloy');
  const [speed, setSpeed] = useState([1.0]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null);

  const generateSpeech = async () => {
    if (!text.trim()) {
      toast.error('Please enter some text to convert to speech');
      return;
    }

    setIsGenerating(true);
    
    try {
      // Use OpenAI TTS for all voices
      const { data, error } = await supabase.functions.invoke('text-to-speech', {
        body: {
          text: text.trim(),
          voice: selectedVoice,
          speed: speed[0],
        },
      });

      if (error) {
        throw new Error(error.message || 'Failed to generate speech');
      }

      if (data.audioContent) {
        // Create audio URL from base64 data
        const audioBlob = new Blob(
          [Uint8Array.from(atob(data.audioContent), c => c.charCodeAt(0))],
          { type: 'audio/mpeg' }
        );
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
        toast.success('Speech generated successfully!');
      }
    } catch (error) {
      console.error('TTS error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to generate speech');
    } finally {
      setIsGenerating(false);
    }
  };

  const playAudio = () => {
    if (!audioUrl) return;

    if (currentAudio) {
      currentAudio.pause();
      setCurrentAudio(null);
    }

    const audio = new Audio(audioUrl);
    audio.onplay = () => setIsPlaying(true);
    audio.onpause = () => setIsPlaying(false);
    audio.onended = () => {
      setIsPlaying(false);
      setCurrentAudio(null);
    };
    
    setCurrentAudio(audio);
    audio.play();
  };

  const pauseAudio = () => {
    if (currentAudio) {
      currentAudio.pause();
    }
  };

  const downloadAudio = () => {
    if (!audioUrl) return;

    const link = document.createElement('a');
    link.href = audioUrl;
    link.download = `tts-audio-${Date.now()}.mp3`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Text-to-Speech Generator</h1>
        <p className="text-muted-foreground">
          Convert your text into natural-sounding speech using advanced AI models
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Input Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Volume2 className="h-5 w-5" />
              Text Input
            </CardTitle>
            <CardDescription>
              Enter the text you want to convert to speech
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="text-input">Text</Label>
              <Textarea
                id="text-input"
                placeholder="Enter your text here..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={6}
                className="resize-none"
              />
              <p className="text-sm text-muted-foreground">
                {text.length} characters
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Settings Section */}
        <Card>
          <CardHeader>
            <CardTitle>Voice Settings</CardTitle>
            <CardDescription>
              Customize the voice and speech parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="voice-select">Voice</Label>
              <Select value={selectedVoice} onValueChange={setSelectedVoice}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a voice" />
                </SelectTrigger>
                <SelectContent>
                  {voices.map((voice) => (
                    <SelectItem key={voice.id} value={voice.id}>
                      <div>
                        <div className="font-medium">{voice.name}</div>
                        <div className="text-sm text-muted-foreground">
                          {voice.description}
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label htmlFor="speed-slider">Speed</Label>
                <span className="text-sm text-muted-foreground">
                  {speed[0].toFixed(1)}x
                </span>
              </div>
              <Slider
                id="speed-slider"
                value={speed}
                onValueChange={setSpeed}
                min={0.25}
                max={4.0}
                step={0.25}
                className="w-full"
              />
            </div>

            <Button 
              onClick={generateSpeech}
              disabled={isGenerating || !text.trim()}
              className="w-full"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating Speech...
                </>
              ) : (
                <>
                  <Volume2 className="mr-2 h-4 w-4" />
                  Generate Speech
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Audio Player Section */}
      {audioUrl && (
        <Card>
          <CardHeader>
            <CardTitle>Generated Audio</CardTitle>
            <CardDescription>
              Play, pause, or download your generated speech
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <Button
                variant="outline"
                size="icon"
                onClick={isPlaying ? pauseAudio : playAudio}
              >
                {isPlaying ? (
                  <Pause className="h-4 w-4" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
              </Button>
              
              <Button
                variant="outline"
                onClick={downloadAudio}
                className="flex-1"
              >
                <Download className="mr-2 h-4 w-4" />
                Download MP3
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}