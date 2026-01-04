'use client';
import { useState } from 'react';
import {
  Box,
  Paper,
  TextField,
  Typography,
  Container,
  LinearProgress,
  Button,
  IconButton,
  Grow,
  TableContainer,
  Table,
  TableHead,
  TableCell,
  TableRow,
  TableBody
} from '@mui/material';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { materialDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import SearchBar from '@/components/SearchBar/SearchBar';
import { v4 as uuidv4 } from 'uuid';

export default function ChatBot() {
  const [isFocused, setIsFocused] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const [generatedQuery, setGeneratedQuery] = useState('');
  const [finalQuery, setFinalQuery] = useState('');
  const [userPrompt, setUserPrompt] = useState('');
  const [isEditing, setIsEditing] = useState(false);

  const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:5000';

  const query = async (userText) => {
    setIsLoading(true);
    setGeneratedQuery('');
    setFinalQuery('');
    setIsEditing(false);

    try {
      // If predict is in the query => call /predict and show text result
      if (userText.toLowerCase().includes('predict')) {
        const response = await fetch(`${BACKEND}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: userText }),
        });

        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data?.error || 'Predict failed');
        }

        const predictedText = data.result;

        const messageId = uuidv4();
        const newMessage = {
          id: messageId,
          userPrompt: userText,
          generatedQuery: predictedText, // display as text
          result: null,
          isNew: true,
          isPredict: true,
        };

        setMessages((prev) => [...prev, newMessage]);
        setTimeout(() => {
          setMessages((prev) => prev.map((m) => (m.id === messageId ? { ...m, isNew: false } : m)));
        }, 50);

        return;
      }

      // otherwise NL->SQL
      const response = await fetch(`${BACKEND}/getSQL`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userText }),
      });

      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data?.error || 'SQL generation failed');
      }

      const sqlText = data.sql ?? data.result ?? ''; // compat with both formats

      setGeneratedQuery(sqlText);
      setFinalQuery(sqlText);
      setUserPrompt(userText);

    } catch (error) {
      console.error('Error:', error);
      alert(error?.message || 'Something went wrong');
    } finally {
      setIsLoading(false);
    }
  };

  const executeQuery = async () => {
    if (!finalQuery?.trim()) return;

    setIsLoading(true);
    const messageId = uuidv4();

    try {
      const response = await fetch(`${BACKEND}/exSQL`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: finalQuery }),
      });

      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data?.error || 'SQL execution failed');
      }

      const results = data.results;

      const newMessage = {
        id: messageId,
        userPrompt: userPrompt,
        generatedQuery: finalQuery,
        result: results,
        isNew: true,
        isPredict: false,
      };

      setMessages((prev) => [...prev, newMessage]);

    } catch (error) {
      console.error('Error executing:', error);
      alert(error?.message || 'Execution failed');
    } finally {
      setIsLoading(false);
      setGeneratedQuery('');
      setFinalQuery('');
      setTimeout(() => {
        setMessages((prev) => prev.map((m) => (m.id === messageId ? { ...m, isNew: false } : m)));
      }, 50);
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        padding: 2,
        minHeight: '100vh',
        transform: `translate(0, ${isFocused ? '0' : '50vh'})`,
        transition: 'all 0.3s ease-in-out',
      }}
    >
      <Container sx={{ flex: 1, overflowY: 'auto', padding: 2 }}>
        {messages.map((message) => (
          <Container
            sx={{
              opacity: message.isNew ? 0 : 1,
              transform: message.isNew ? 'translateY(-20px)' : 'translateY(0)',
              transition: 'all 0.5s ease-out',
            }}
            key={message.id}
          >
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2, mb: 2 }}>
              <Paper sx={{ p: 2, borderRadius: 2, width: '100%' }}>
                <SyntaxHighlighter language="text" style={materialDark} customStyle={{ width: '100%', borderRadius: '8px', marginBottom: '16px' }}>
                  {message.userPrompt}
                </SyntaxHighlighter>

                <SyntaxHighlighter language={message.isPredict ? "text" : "sql"} style={materialDark} customStyle={{ width: '100%', borderRadius: '8px', marginBottom: '16px' }}>
                  {message.generatedQuery}
                </SyntaxHighlighter>

                {message.result && (
                  <TableContainer component={Paper} sx={{ maxHeight: 440 }}>
                    <Table stickyHeader>
                      <TableHead>
                        <TableRow>
                          {message.result.columns.map((column) => (
                            <TableCell key={column}>{column}</TableCell>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {message.result.rows.map((row, rowIndex) => (
                          <TableRow key={rowIndex}>
                            {row.map((cell, cellIndex) => (
                              <TableCell key={cellIndex}>{String(cell)}</TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </Paper>
            </Box>
          </Container>
        ))}

        {/* SQL Preview + Execute (only if generatedQuery exists and it's SQL) */}
        <Grow in={generatedQuery !== ""}>
          <Box
            sx={{
              display: generatedQuery ? 'flex' : 'none',
              flexDirection: 'column',
              alignItems: 'center',
              p: 2,
              mb: 2,
            }}
          >
            <Box sx={{ width: '100%', display: 'flex', alignItems: 'center' }}>
              {isEditing ? (
                <TextField
                  fullWidth
                  multiline
                  rows={2}
                  value={finalQuery}
                  onChange={(e) => setFinalQuery(e.target.value)}
                  placeholder="Edit the generated SQL query"
                />
              ) : (
                <SyntaxHighlighter language="sql" style={materialDark} customStyle={{ width: "100%", borderRadius: '8px', marginBottom: '16px' }}>
                  {finalQuery}
                </SyntaxHighlighter>
              )}

              <IconButton
                onClick={() => setIsEditing(!isEditing)}
                sx={{ width: "32px", height: "32px", transform: 'translateX(-150%)' }}
              >
                {isEditing ? <SaveIcon /> : <EditIcon />}
              </IconButton>
            </Box>

            <Button
              variant="contained"
              color="primary"
              onClick={executeQuery}
              sx={{ mt: 2 }}
              disabled={isEditing || isLoading}
            >
              Execute Query
            </Button>
          </Box>
        </Grow>

        <Grow in={isLoading}>
          <Box sx={{ m: 2, overflow: 'hidden', display: isLoading ? '' : 'none' }}>
            <Paper sx={{ p: 2, borderRadius: 2, width: '100%' }}>
              <Typography variant="body1">{generatedQuery === "" ? 'Generating...' : 'Executing...'}</Typography>
            </Paper>
            <LinearProgress />
          </Box>
        </Grow>

        <Box
          sx={{
            opacity: isLoading ? 0 : 1,
            transform: isLoading ? 'translateY(-20px)' : 'translateY(0)',
            transition: 'all 0.5s ease-out',
          }}
        >
          <SearchBar
            onFocus={() => setIsFocused(true)}
            onSearch={query}
            disabled={isLoading}
            placeholder={'Ask a question (or include "predict")'}
          />
        </Box>
      </Container>
    </Box>
  );
}
