FROM node:18-alpine

WORKDIR /app

# Copy package files first for better caching
COPY package.json package-lock.json* ./

# Install dependencies (include devDependencies)
RUN npm install

# Copy the rest of the files
COPY . .

# Environment variables for development
ENV NODE_ENV=development
ENV NEXT_TELEMETRY_DISABLED=1

# Expose ports (Next.js + debugger)
EXPOSE 3000

# Start Next.js in dev mode with debugger
CMD ["npm", "run", "dev"]
