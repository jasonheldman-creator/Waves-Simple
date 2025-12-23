import { NextRequest, NextResponse } from "next/server";

interface ContactFormData {
  name: string;
  email: string;
  company: string;
  message: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: ContactFormData = await request.json();

    // Validate required fields
    const { name, email, company, message } = body;

    if (!name || !email || !company || !message) {
      return NextResponse.json({ error: "All fields are required" }, { status: 400 });
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return NextResponse.json({ error: "Invalid email format" }, { status: 400 });
    }

    // Validate field lengths
    if (name.length < 2 || name.length > 100) {
      return NextResponse.json(
        { error: "Name must be between 2 and 100 characters" },
        { status: 400 }
      );
    }

    if (company.length < 2 || company.length > 100) {
      return NextResponse.json(
        { error: "Company name must be between 2 and 100 characters" },
        { status: 400 }
      );
    }

    if (message.length < 10 || message.length > 5000) {
      return NextResponse.json(
        { error: "Message must be between 10 and 5000 characters" },
        { status: 400 }
      );
    }

    // Log the submitted data server-side
    console.log("=== Contact Form Submission ===");
    console.log("Timestamp:", new Date().toISOString());
    console.log("Name:", name);
    console.log("Email:", email);
    console.log("Company:", company);
    console.log("Message:", message);
    console.log("================================");

    // Return success response
    return NextResponse.json(
      {
        success: true,
        message: "Your message has been received. We'll be in touch soon!",
      },
      { status: 200 }
    );
  } catch {
    console.error("Error processing contact form");
    return NextResponse.json(
      { error: "An error occurred processing your request" },
      { status: 500 }
    );
  }
}
