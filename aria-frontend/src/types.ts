export interface TextRegion {
  id: string;
  source_text: string;
  translated_text: string;
  confidence: number;
  reading_order: number;
}

export interface ManualInpaintRegion {
  bbox: [number, number, number, number];
}

export interface PageStatus {
  id: string;
  filename: string;
  status: 'queued' | 'processing' | 'complete' | 'failed';
  original_url: string;
  cleaned_url: string | null;
  output_url: string | null;
  regions: TextRegion[];
  manual_inpaint_regions: ManualInpaintRegion[];
  warnings: string[];
  error: string | null;
}

export interface JobStatus {
  id: string;
  status: 'queued' | 'processing' | 'complete' | 'failed';
  pages: PageStatus[];
  error: string | null;
}
