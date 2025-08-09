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
    const { text, max_tokens, temperature, top_p } = await req.json()

    if (!text) {
      throw new Error('Text is required')
    }

    console.log(`Attempting to use Dia model directly for text: ${text}`)

    // Use Python subprocess to run Dia model locally
    // This would require the model to be installed on the edge function runtime
    // which isn't currently supported by Supabase Edge Functions
    
    // Alternative: Call your deployed Hugging Face Space directly
    const spaceUrl = 'https://chrishugs-dia-tts-nari.hf.space'
    
    const response = await fetch(`${spaceUrl}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        data: [text, max_tokens || 3072, temperature || 0.7, top_p || 0.9]
      }),
    })

    if (!response.ok) {
      throw new Error(`HF Space API error: ${response.status}`)
    }

    const data = await response.json()
    
    // Extract audio from response
    if (data.data && data.data[0]) {
      return new Response(
        JSON.stringify({
          audio_content: data.data[0], // Base64 audio
          message: `Generated speech using Dia model`
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    throw new Error('No audio generated from Dia model')

  } catch (error) {
    console.error('Error in dia-direct-load function:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    )
  }
})