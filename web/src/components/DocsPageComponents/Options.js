import React from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { solarizedLight } from 'react-syntax-highlighter/dist/esm/styles/hljs';


export const GeneralOptions = {
  _k: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// key of data\n-k [ --key ] arg'}
    </SyntaxHighlighter>
  ),
  _b: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// the operating branch\n-b [ --branch ] arg'}
    </SyntaxHighlighter>
  ),
  _v: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// the operating version\n-v [ --version ] arg'}
    </SyntaxHighlighter>
  ),
  _file: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {`// path to the file on system:\nfile`}
    </SyntaxHighlighter>
  ),
  _x: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// data value\n-x [ --value ] arg'}
    </SyntaxHighlighter>
  ),
  _p: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// data value\n-p [ --type ] arg (=Blob)'}
    </SyntaxHighlighter>
  ),
  _u: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// the referring version\n-u [ --ref-version ] arg'}
    </SyntaxHighlighter>
  ),
  _i: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// the operating positional index\n-i [ --position ] arg'}
    </SyntaxHighlighter>
  ),
  _e: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// key of map entry\n-e [ --map-key ] arg'}
    </SyntaxHighlighter>
  ),
  _d: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// number of elements\n-d [ --num-elements ] arg (=1)'}
    </SyntaxHighlighter>
  ),
  _c: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// the referring branch\n-c [ --ref-branch ] arg'}
    </SyntaxHighlighter>
  ),
  _none: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// none'}
    </SyntaxHighlighter>
  ),
}


export const UtilityOptions = {
  _none: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// none'}
    </SyntaxHighlighter>
  ),
  _1: (
    <SyntaxHighlighter language='javascript' style={solarizedLight}>
      {'// (it is one, not "l") list one entry per line \n-1 [ --vert-list ]'}
    </SyntaxHighlighter>
  ),
}
