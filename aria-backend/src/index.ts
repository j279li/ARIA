export interface Env {
	ARIA_STORAGE: R2Bucket;
	GOOGLE_VISION_API_KEY: string;
	DEEPL_API_KEY: string;
  }
  
  interface OCRResult {
	text: string;
	box: [number, number, number, number]; // [ymin, xmin, ymax, xmax]
  }

  // Helper function to handle retranslation requests
  async function handleRetranslate(request: Request, env: Env, corsHeaders: any): Promise<Response> {
	try {
	  const body = await request.json() as any;
	  const textsToTranslate = body.texts as string[];
	  
	  if (!textsToTranslate || !Array.isArray(textsToTranslate) || textsToTranslate.length === 0) {
		return new Response(JSON.stringify({ error: "Missing or invalid texts array" }), { 
		  status: 400, 
		  headers: { ...corsHeaders, "Content-Type": "application/json" }
		});
	  }

	  console.log(`Retranslating ${textsToTranslate.length} segments with new order`);

	  // Join all text segments with separator for context-aware translation
	  const SEPARATOR = '\n<<<SEP>>>\n';
	  const combinedText = textsToTranslate.join(SEPARATOR);
	  
	  const deeplApiUrl = `https://api-free.deepl.com/v2/translate`;
	  const deeplResponse = await fetch(deeplApiUrl, {
		method: 'POST',
		headers: { 
		  'Authorization': `DeepL-Auth-Key ${env.DEEPL_API_KEY}`,
		  'Content-Type': 'application/json'
		},
		body: JSON.stringify({
		  text: [combinedText],
		  source_lang: 'JA',
		  target_lang: 'EN-US',
		  formality: 'default',
		  preserve_formatting: true
		})
	  });

	  if (!deeplResponse.ok) {
		const errorText = await deeplResponse.text();
		console.error("DeepL API error:", errorText);
		throw new Error(`DeepL API failed: ${deeplResponse.status} ${errorText}`);
	  }

	  const deeplResult = await deeplResponse.json() as any;
	  const translatedText = deeplResult.translations[0].text;
	  
	  // Split translated text back into individual segments
	  const translatedSegments = translatedText.split(SEPARATOR);
	  
	  // Handle case where split count doesn't match
	  if (translatedSegments.length !== textsToTranslate.length) {
		console.warn(`Segment mismatch: expected ${textsToTranslate.length}, got ${translatedSegments.length}`);
		while (translatedSegments.length < textsToTranslate.length) {
		  translatedSegments.push('[Translation error]');
		}
	  }

	  return Response.json({ 
		translations: translatedSegments.map((t: string) => t.trim())
	  }, { headers: corsHeaders });

	} catch (error: any) {
	  console.error("Retranslation error:", error);
	  return new Response(
		JSON.stringify({ 
		  error: "Retranslation failed", 
		  details: error?.message || "Unknown error"
		}), 
		{ 
		  status: 500, 
		  headers: { ...corsHeaders, "Content-Type": "application/json" }
		}
	  );
	}
  }
  
  export default {
	async fetch(request: Request, env: Env): Promise<Response> {
	  const corsHeaders = {
		"Access-Control-Allow-Origin": "*",
		"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
		"Access-Control-Allow-Headers": "Content-Type",
	  };
  
	  // 1. Handle the "Preflight" request from the browser
	  if (request.method === "OPTIONS") {
		return new Response(null, { headers: corsHeaders });
	  }

	  if (request.method !== "POST") {
		return new Response("Please POST an image or translation request", { status: 405, headers: corsHeaders });
	  }

	  const url = new URL(request.url);
	  
	  // Handle retranslation endpoint
	  if (url.pathname === '/retranslate') {
		return handleRetranslate(request, env, corsHeaders);
	  }
  
	  try {
		const formData = await request.formData();
		const imageFile = formData.get("image") as File;
		
		if (!imageFile) {
		  return new Response(JSON.stringify({ error: "Missing image file" }), { 
			status: 400, 
			headers: { ...corsHeaders, "Content-Type": "application/json" }
		  });
		}
		
		console.log("Processing image:", imageFile.name, imageFile.type);
		const blob = await imageFile.arrayBuffer();
		console.log("Image size:", blob.byteLength, "bytes");

		// 2. OCR Step - Use Google Cloud Vision API
		console.log("Running Google Cloud Vision OCR...");
		
		// Convert ArrayBuffer to base64 without stack overflow
		const uint8Array = new Uint8Array(blob);
		let binary = '';
		const chunkSize = 8192;
		for (let i = 0; i < uint8Array.length; i += chunkSize) {
		  const chunk = uint8Array.subarray(i, i + chunkSize);
		  binary += String.fromCharCode.apply(null, Array.from(chunk));
		}
		const base64Image = btoa(binary);
		
		const visionApiUrl = `https://vision.googleapis.com/v1/images:annotate?key=${env.GOOGLE_VISION_API_KEY}`;
		const visionResponse = await fetch(visionApiUrl, {
		  method: 'POST',
		  headers: { 'Content-Type': 'application/json' },
		  body: JSON.stringify({
			requests: [{
			  image: { content: base64Image },
			  features: [{ type: 'TEXT_DETECTION' }],
			  imageContext: { languageHints: ['ja'] }
			}]
		  })
		});

		if (!visionResponse.ok) {
		  const errorText = await visionResponse.text();
		  console.error("Google Vision API error:", errorText);
		  throw new Error(`Google Vision API failed: ${visionResponse.status} ${errorText}`);
		}

		const visionResult = await visionResponse.json() as any;
		console.log("Vision result:", JSON.stringify(visionResult));
		
		// 3. Extract text blocks from Google Vision response
		const annotations = visionResult.responses?.[0]?.textAnnotations || [];
		
		if (annotations.length === 0) {
		  return Response.json({ 
			results: [], 
			debug: {
			  message: "No text detected in the image.",
			  suggestion: "Try a higher quality image or one with clearer text."
			}
		  }, { headers: corsHeaders });
		}
		
		// Skip first annotation (it's the full text), use individual text blocks
		const textBlocks = annotations.slice(1);
		console.log(`Detected ${textBlocks.length} text blocks`);
		
		// Helper function to calculate distance between two boxes
		interface Box {
		  xmin: number;
		  ymin: number;
		  xmax: number;
		  ymax: number;
		  text: string;
		  grouped: boolean;
		}
		
		interface BoundingBox {
		  xmin: number;
		  ymin: number;
		  xmax: number;
		  ymax: number;
		}
		
		const boxDistance = (box1: BoundingBox, box2: BoundingBox) => {
		  const centerX1 = (box1.xmin + box1.xmax) / 2;
		  const centerY1 = (box1.ymin + box1.ymax) / 2;
		  const centerX2 = (box2.xmin + box2.xmax) / 2;
		  const centerY2 = (box2.ymin + box2.ymax) / 2;
		  return Math.sqrt(Math.pow(centerX2 - centerX1, 2) + Math.pow(centerY2 - centerY1, 2));
		};
		
		// Convert annotations to simple format
		const boxes: Box[] = textBlocks.map((annotation: any) => {
		  const vertices = annotation.boundingPoly.vertices;
		  const xCoords = vertices.map((v: any) => v.x || 0);
		  const yCoords = vertices.map((v: any) => v.y || 0);
		  
		  return {
			text: annotation.description,
			xmin: Math.min(...xCoords),
			ymin: Math.min(...yCoords),
			xmax: Math.max(...xCoords),
			ymax: Math.max(...yCoords),
			grouped: false
		  };
		});
		
		// Calculate adaptive threshold based on image size and text box sizes
		const imageWidth = Math.max(...boxes.map(b => b.xmax));
		const imageHeight = Math.max(...boxes.map(b => b.ymax));
		const imageDiagonal = Math.sqrt(imageWidth * imageWidth + imageHeight * imageHeight);
		
		// Average text box size
		const avgBoxWidth = boxes.reduce((sum, b) => sum + (b.xmax - b.xmin), 0) / boxes.length;
		const avgBoxHeight = boxes.reduce((sum, b) => sum + (b.ymax - b.ymin), 0) / boxes.length;
		const avgBoxSize = Math.sqrt(avgBoxWidth * avgBoxWidth + avgBoxHeight * avgBoxHeight);
		
		// Dynamic threshold: 3x average box size, or 2% of image diagonal, whichever is larger
		const threshold = Math.max(avgBoxSize * 3, imageDiagonal * 0.02);
		console.log(`Using grouping threshold: ${threshold.toFixed(0)}px (image: ${imageWidth}x${imageHeight}, avg box: ${avgBoxSize.toFixed(0)}px)`);
		
		// Group nearby text blocks (speech bubbles)
		const groups: any[] = [];
		
		boxes.forEach(box => {
		  if (box.grouped) return;
		  
		  // Start a new group
		  const group = {
			texts: [box.text],
			xmin: box.xmin,
			ymin: box.ymin,
			xmax: box.xmax,
			ymax: box.ymax
		  };
		  box.grouped = true;
		  
		  // Find nearby boxes to add to this group
		  let changed = true;
		  while (changed) {
			changed = false;
			boxes.forEach(otherBox => {
			  if (otherBox.grouped) return;
			  
			  // Check if this box is close to the group
			  const dist = boxDistance(group, otherBox);
			  if (dist < threshold) {
				group.texts.push(otherBox.text);
				group.xmin = Math.min(group.xmin, otherBox.xmin);
				group.ymin = Math.min(group.ymin, otherBox.ymin);
				group.xmax = Math.max(group.xmax, otherBox.xmax);
				group.ymax = Math.max(group.ymax, otherBox.ymax);
				otherBox.grouped = true;
				changed = true;
			  }
			});
		  }
		  
		  groups.push(group);
		});
		
	console.log(`Grouped into ${groups.length} speech bubbles`);
	
	// Calculate centers for each group
	groups.forEach((group: any) => {
	  group.centerX = (group.xmin + group.xmax) / 2;
	  group.centerY = (group.ymin + group.ymax) / 2;
	});
	
	// Manga reading order: Right-to-left, top-to-bottom
	// Higher threshold (2x) because manga panels are large
	groups.sort((a: any, b: any) => {
	  const yDiff = a.centerY - b.centerY; // Top-to-bottom
	  const xDiff = b.centerX - a.centerX; // Right-to-left
	  
	  // If at significantly different Y levels, sort by Y first
	  if (Math.abs(yDiff) > avgBoxHeight * 2.0) {
		return yDiff;
	  }
	  
	  // Same row/panel, sort right-to-left
	  return xDiff;
	});
	
	console.log('Bubble reading order:', groups.map((g: any, i: number) => 
	  `${i+1}: x=${g.centerX.toFixed(0)} y=${g.centerY.toFixed(0)}`
	).join(', '));
		
		// Convert groups to detectedText format
		const detectedText = groups.map(group => ({
		  text: group.texts.join(''),
		  box: [group.ymin, group.xmin, group.ymax, group.xmax]
		}));
  
	// 4. Context-Aware Translation using DeepL
	console.log("Starting context-aware translation with DeepL...");
	
	// Join all text segments with a special separator to maintain context
	// This allows DeepL to see the entire page conversation flow
	const SEPARATOR = '\n<<<SEP>>>\n';
	const textsToTranslate = detectedText.map((item: any) => item.text);
	const combinedText = textsToTranslate.join(SEPARATOR);
	
	console.log(`Translating ${textsToTranslate.length} segments as one context block`);
	
	const deeplApiUrl = `https://api-free.deepl.com/v2/translate`;
	const deeplResponse = await fetch(deeplApiUrl, {
	  method: 'POST',
	  headers: { 
		'Authorization': `DeepL-Auth-Key ${env.DEEPL_API_KEY}`,
		'Content-Type': 'application/json'
	  },
	  body: JSON.stringify({
		text: [combinedText], // DeepL expects an array of strings
		source_lang: 'JA',
		target_lang: 'EN-US',
		formality: 'default',
		preserve_formatting: true
	  })
	});

	if (!deeplResponse.ok) {
	  const errorText = await deeplResponse.text();
	  console.error("DeepL API error:", errorText);
	  throw new Error(`DeepL API failed: ${deeplResponse.status} ${errorText}`);
	}

	const deeplResult = await deeplResponse.json() as any;
	const translatedText = deeplResult.translations[0].text;
	console.log("Translation complete with full context");
	
	// Split translated text back into individual segments
	const translatedSegments = translatedText.split(SEPARATOR);
	
	// Handle case where split count doesn't match (fallback)
	if (translatedSegments.length !== textsToTranslate.length) {
	  console.warn(`Segment mismatch: expected ${textsToTranslate.length}, got ${translatedSegments.length}`);
	  // Pad or truncate to match
	  while (translatedSegments.length < textsToTranslate.length) {
		translatedSegments.push('[Translation error]');
	  }
	}
	
	// Combine original text with translations
	const results = detectedText.map((item: any, index: number) => ({
	  ...item,
	  translatedText: translatedSegments[index]?.trim() || '[Translation error]'
	}));
  
		// 5. Return with CORS headers
		return Response.json({ results }, { headers: corsHeaders });
	  } catch (error: any) {
		console.error("Worker error:", error);
		console.error("Error stack:", error?.stack);
		return new Response(
		  JSON.stringify({ 
			error: "Processing failed", 
			details: error?.message || "Unknown error",
			stack: error?.stack || ""
		  }), 
		  { 
			status: 500, 
			headers: { ...corsHeaders, "Content-Type": "application/json" }
		  }
		);
	  }
	},
  };