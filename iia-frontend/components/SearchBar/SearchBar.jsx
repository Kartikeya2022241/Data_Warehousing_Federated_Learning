// SearchBar.jsx
import React, { useState } from 'react';
import { Box, IconButton, InputBase, Paper } from '@mui/material';
import { Search as SearchIcon, Clear as ClearIcon } from '@mui/icons-material';

const SearchBar = ({ onSearch, onFocus,onBlur,disabled,value,placeholder }) => {
  const [query, setQuery] = useState(value ? value : '');
  const handleSearch = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query);
      setQuery('');
    }
  };

  const handleClear = () => {
    setQuery('');
  };

  return (
    <Box
      component="form"
      onSubmit={handleSearch}
      sx={{
        width: '100%',
        maxWidth: 600,
        mx: 'auto'
      }}
    >
      <Paper
        elevation={1}
        sx={{
          p: '2px 4px',
          display: 'flex',
          alignItems: 'center',
          borderRadius: 100,
          border: '1px solid transparent',
          '&:hover': {
            boxShadow: '0 1px 6px rgb(32 33 36 / 28%)',
            borderColor: 'rgba(223,225,229,0)'
          }
        }}
      >
        <IconButton sx={{ p: '10px' }}>
          <SearchIcon sx={{ color: '#5f6368' }} />
        </IconButton>
        <InputBase
          sx={{ ml: 1, flex: 1 }}
          placeholder={placeholder}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus = {onFocus}
          onBlur={onBlur}
          disabled={disabled}
        />
        {query && (
          <IconButton onClick={handleClear} sx={{ p: '10px' }}>
            <ClearIcon sx={{ color: '#5f6368' }} />
          </IconButton>
        )}
      </Paper>
    </Box>
  );
};

export default SearchBar;