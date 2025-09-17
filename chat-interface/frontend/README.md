# Spectacular Chat Interface Frontend

Beautiful, modern React/Next.js frontend for the Spectacular Triple Validation Framework with Bayesian Evidence Network.

## ✨ Features

### 🎨 Beautiful Modern UI

- **Glass morphism design** with gradient backgrounds
- **Responsive layout** that works on all screen sizes
- **Animated components** with smooth transitions
- **Dark theme** optimized for data visualization
- **Real-time status indicators** and loading states

### 🧠 Bayesian Network Integration

- **Real-time processing status** with network health monitoring
- **Triple validation plot display** (Pugachev-Cobra, Intent, Reasoning)
- **Environmental sensor data visualization**
- **Dynamic routing and recursion indicators**
- **Multi-dimensional embedding path tracking**

### 📊 Advanced Data Visualization

- **SVG plot rendering** with interactive elements
- **Confidence scoring** with visual indicators
- **Processing time tracking** and performance metrics
- **Validation details** with expandable metadata
- **System status dashboard** with component health

### 🚀 Performance Optimized

- **Next.js 14** with App Router
- **TypeScript** for type safety
- **Tailwind CSS** for efficient styling
- **Axios** with request/response interceptors
- **Toast notifications** for user feedback

## 🏗️ Architecture

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── ChatInterface.tsx    # Main chat interface
│   │   ├── PlotDisplay.tsx      # Triple validation plots
│   │   └── SystemStatus.tsx     # System health dashboard
│   ├── lib/
│   │   └── api.ts              # API client with full typing
│   └── types/
│       └── api.ts              # TypeScript definitions
├── pages/
│   ├── _app.tsx               # App configuration
│   ├── _document.tsx          # HTML document structure
│   └── index.tsx              # Main page
└── styles/
    └── globals.css            # Global styles with glass effects
```

## 🔧 API Integration

The frontend integrates seamlessly with the FastAPI backend:

### Chat Endpoint

```typescript
POST /api/chat
{
  message: string;
  context?: Record<string, any>;
  conversation_id?: string;
}
```

**Response includes:**

- AI-generated response text
- Triple validation plots (SVG content)
- Bayesian network processing details
- Environmental sensor data
- Performance metrics

### System Status

```typescript
GET / api / status;
```

**Real-time monitoring of:**

- Bayesian network health
- Environmental sensor status
- Pipeline component status
- Processing performance metrics

## 🎯 User Experience Flow

### 1. **Query Submission**

- User types query in elegant chat input
- Auto-resizing textarea with keyboard shortcuts
- Real-time validation and error handling

### 2. **Processing Visualization**

- Animated loading states with Bayesian network branding
- Real-time status updates during processing
- Environmental sensor integration indicators

### 3. **Results Display**

- AI response with markdown support
- Triple validation plots in responsive grid
- Processing summary with metrics
- Expandable validation details

### 4. **System Monitoring**

- Collapsible system status sidebar
- Real-time health monitoring
- Performance metrics tracking
- Component status indicators

## 🎨 Design System

### Colors

- **Primary**: Blue gradients for AI/neural elements
- **Secondary**: Purple/pink for Bayesian network elements
- **Success**: Green for validation and health
- **Warning**: Yellow/orange for alerts
- **Error**: Red for failures

### Components

- **Glass effect**: `backdrop-blur-md bg-white/10`
- **Neural glow**: Blue shadows for AI elements
- **Bayesian glow**: Purple shadows for network elements
- **Smooth animations**: Consistent 200-300ms transitions

### Typography

- **Headers**: Inter font family, bold weights
- **Body**: Inter font family, regular weights
- **Code/Data**: JetBrains Mono for metrics and data

## 🔌 Environment Configuration

Create `.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

For production:

```env
NEXT_PUBLIC_API_BASE_URL=https://api.spectacular.ai
```

## 🚀 Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## 📱 Responsive Design

- **Mobile**: Single column layout with collapsible sidebar
- **Tablet**: Adaptive grid for plots, optimized touch targets
- **Desktop**: Full multi-column layout with persistent sidebar
- **Large screens**: Maximum utilization of space for data visualization

## 🔒 Security

- **Content Security Policy** headers configured
- **CORS** properly configured for API communication
- **Input validation** on all user inputs
- **Error boundaries** for graceful error handling

## 🎛️ Keyboard Shortcuts

- **Ctrl/Cmd + K**: Focus chat input
- **Enter**: Send message
- **Shift + Enter**: New line in message
- **Esc**: Close modals/sidebar

## 📊 Performance Metrics

The interface tracks and displays:

- **Processing time** for each query
- **Network coherence** from Bayesian network
- **Environmental sensor status**
- **System resource utilization**
- **API response times**

## 🌟 Key Components Explained

### PlotDisplay Component

- Renders SVG plots from the API
- Shows confidence scores with color coding
- Expandable metadata for analysis details
- Loading states during plot generation

### ChatInterface Component

- Manages chat state and message history
- Integrates with Bayesian network API
- Shows processing status and validation results
- Auto-scrolling and message persistence

### SystemStatus Component

- Real-time system health monitoring
- Environmental sensor data display
- Bayesian network node status
- Performance metrics and uptime

This frontend provides a beautiful, modern interface for interacting with the sophisticated Bayesian Evidence Network backend, making complex AI reasoning accessible and visually compelling.
