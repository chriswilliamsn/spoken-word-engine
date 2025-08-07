import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { text, max_tokens, temperature, top_p } = await req.json()

    if (!text) {
      throw new Error('Text is required')
    }

    console.log('Generating audio with Dia TTS for text:', text)

    // Get the Dia TTS server URL from environment
    const DIA_SERVER_URL = Deno.env.get('DIA_SERVER_URL') || 'http://localhost:8000'
    
    // First check if the server is healthy
    console.log(`Checking health of Dia server at: ${DIA_SERVER_URL}`)
    
    try {
      const healthResponse = await fetch(`${DIA_SERVER_URL}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      })
      
      if (!healthResponse.ok) {
        throw new Error(`Health check failed: ${healthResponse.status} ${healthResponse.statusText}`)
      }
      
      const healthData = await healthResponse.json()
      console.log('Health check response:', healthData)
      
      if (!healthData.model_loaded) {
        throw new Error('Dia model is not loaded on the server')
      }
    } catch (healthError) {
      console.error('Health check failed:', healthError)
      throw new Error(`Dia server is not available: ${healthError.message}`)
    }

    // Call the Python Dia TTS server
    console.log(`Calling Dia TTS server at: ${DIA_SERVER_URL}/generate`)
    const response = await fetch(`${DIA_SERVER_URL}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: text.trim(),
        max_tokens: max_tokens || 3072,
        temperature: temperature || 0.7,
        top_p: top_p || 0.9,
      }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(`Dia TTS server error: ${errorData.detail || response.statusText}`)
    }

    const data = await response.json()
    console.log('Dia TTS generation complete')

    return new Response(
      JSON.stringify({
        audioContent: data.audio_content,
        message: data.message,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    )
  } catch (error) {
    console.error('Error in dia-tts function:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      },
    )
  }
})