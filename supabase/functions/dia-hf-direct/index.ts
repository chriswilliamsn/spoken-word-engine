import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    let requestBody;
    try {
      const rawBody = await req.text();
      if (!rawBody || rawBody.trim() === '') {
        throw new Error('Request body is empty');
      }
      requestBody = JSON.parse(rawBody);
    } catch (parseError) {
      console.error('Failed to parse request body:', parseError);
      return new Response(
        JSON.stringify({ error: 'Invalid JSON in request body' }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    const { text, max_tokens, temperature, top_p } = requestBody;

    if (!text) {
      throw new Error('Text is required')
    }

    console.log(`Generating audio with Hugging Face API for text: ${text}`);

    // Use a TTS model available via HF Inference API
    const hfToken = Deno.env.get('HUGGING_FACE_TOKEN');
    if (!hfToken) {
      throw new Error('Hugging Face token not configured');
    }

    // Using Microsoft SpeechT5 TTS model which is available via Inference API
    const response = await fetch(
      'https://api-inference.huggingface.co/models/microsoft/speecht5_tts',
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${hfToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          inputs: text,
        }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Hugging Face API error:', errorText);
      throw new Error(`Hugging Face API error: ${response.status} ${errorText}`);
    }

    // Get audio buffer
    const audioBuffer = await response.arrayBuffer();
    const audioBase64 = btoa(String.fromCharCode(...new Uint8Array(audioBuffer)));

    console.log('Audio generation completed successfully');

    return new Response(
      JSON.stringify({
        audio_content: audioBase64,
        message: `Generated speech for: "${text.substring(0, 50)}..."`
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Error in dia-hf-direct function:', error);
    return new Response(
      JSON.stringify({ error: error.message }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );
  }
});